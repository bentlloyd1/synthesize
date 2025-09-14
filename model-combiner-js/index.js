import express from 'express';
import OpenAI from 'openai';
import { GoogleGenerativeAI } from '@google/generative-ai';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config();

const app = express();
const port = 3000;

app.use(express.json());
app.use(express.static(path.resolve()));

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// --- CONFIGURATION ---
const CLASSIFIER_MODEL = 'gpt-4o';
const DYNAMIC_PIPELINES = {
    "FACTUAL": {
        name: "Factual Pipeline (SOTA Base -> Quick Synth)",
        config: { gpt: 'gpt-5', gemini: 'gemini-2.5-pro', synthesizer: 'gpt-4o' }
    },
    "CREATIVE": {
        name: "Creative Pipeline (SOTA Base -> SOTA Synth)",
        config: { gpt: 'gpt-5', gemini: 'gemini-2.5-pro', synthesizer: 'gemini-2.5-pro' }
    }
};

// --- HELPER to format chat history ---
function formatChatHistory(history) {
    if (!history || history.length === 0) return "No previous conversation.";
    return history.map(turn => `${turn.role === 'user' ? 'User' : 'Assistant'}: ${turn.content}`).join('\n');
}

// --- META-PROMPTS (Updated to accept chat history) ---
function getSynthesisPrompt(prompt, gptResponse, geminiResponse, constraints = '', chatHistory = []) {
    const historyText = formatChatHistory(chatHistory);
    const constraintInstruction = constraints ? `\n\nCRITICAL INSTRUCTION: Adhere to this constraint: ${constraints}\n` : '';

    return `You are an expert synthesizer. Given a conversation history and two new draft responses to the user's latest prompt, your task is to merge them into one superior, cohesive response.

Conversation History:
${historyText}

User's Latest Prompt: "${prompt}"

---
Draft Response A:
"${gptResponse}"

---
Draft Response B:
"${geminiResponse}"
---
${constraintInstruction}
Synthesize the two drafts into a single, final response that directly and thoughtfully answers the user's latest prompt, maintaining the context of the conversation. Final Response:`;
}

function getRefinementPrompt(prompt, singleResponse, constraints = '', chatHistory = []) {
     const historyText = formatChatHistory(chatHistory);
     const constraintInstruction = constraints ? `\n\nCRITICAL INSTRUCTION: Adhere to this constraint: ${constraints}\n` : '';

    return `You are an expert editor. Review the following draft response in the context of the conversation history and the user's latest prompt. Your task is to refine and improve it.

Conversation History:
${historyText}

User's Latest Prompt: "${prompt}"

---
Draft Response to Refine:
"${singleResponse}"
---
${constraintInstruction}
Rewrite the draft to be a superior, final response that directly answers the user's latest prompt and adheres to any constraints. Final Response:`;
}

function getClassificationPrompt(prompt, chatHistory = []) {
    const historyText = formatChatHistory(chatHistory);
    return `Analyze the user's LATEST prompt in the context of the conversation history. Classify the intent of the LATEST prompt as "FACTUAL" or "CREATIVE".

- "FACTUAL": Requests for explanations, code, technical info, summaries.
- "CREATIVE": Requests for stories, brainstorming, poetry, open-ended tasks.

First, provide a brief reasoning. Second, on a new line, provide the classification.

Conversation History:
${historyText}

User's Latest Prompt: "${prompt}"

Reasoning:
Classification:`;
}

// --- API CALLS ---
async function getGptResponse(prompt, modelName, chatHistoryForApi = []) {
    try {
        const messages = [...chatHistoryForApi, { role: "user", content: prompt }];
        const response = await openai.chat.completions.create({ model: modelName, messages: messages });
        return response.choices[0].message.content;
    } catch (error) {
        return `ERROR: OpenAI API call failed for model ${modelName}.`;
    }
}
async function getGeminiResponse(prompt, modelName, chatHistoryForApi = []) {
    try {
        const model = genAI.getGenerativeModel({ model: modelName });
        const chat = model.startChat({ history: chatHistoryForApi });
        const result = await chat.sendMessage(prompt);
        return result.response.text();
    } catch (error) {
        return `ERROR: Google AI API call failed for model ${modelName}.`;
    }
}

// Helper to convert our simple history to API-specific formats
function convertHistoryForApi(history, apiType) {
    if (!history) return [];
    if (apiType === 'openai') {
        return history.map(turn => ({ role: turn.role, content: turn.content }));
    }
    if (apiType === 'gemini') {
        return history.map(turn => ({ role: turn.role, parts: [{ text: turn.content }] }));
    }
    return [];
}


// --- THE /combine ENDPOINT ---
app.post('/combine', async (req, res) => {
    const { prompt, constraints, chatHistory } = req.body;
    if (!prompt) return res.status(400).json({ error: "Prompt is required." });

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const sendEvent = (event, data) => {
        res.write(`event: ${event}\n`);
        res.write(`data: ${JSON.stringify(data)}\n\n`);
    };

    try {
        const historyForPrompting = chatHistory.slice(0, -1); // History excluding the current prompt

        // Step 1: Classify intent
        sendEvent('status', { message: "Classifying prompt intent..." });
        const classificationResponse = await getGptResponse(getClassificationPrompt(prompt, historyForPrompting), CLASSIFIER_MODEL);
        const classifierLines = classificationResponse.trim().split('\n');
        const finalClassification = classifierLines.pop().trim().toUpperCase();
        const reasoning = classifierLines.join(' ').replace('Reasoning:', '').trim();
        const intent = finalClassification.includes("CREATIVE") ? "CREATIVE" : "FACTUAL";
        const pipeline = DYNAMIC_PIPELINES[intent];
        const config = pipeline.config;
        
        sendEvent('initial-data', { pipelineName: pipeline.name, classifierReasoning: reasoning });
        sendEvent('status', { message: "Generating base responses..." });
        
        // Step 2: Base Generation
        const gptHistoryForApi = convertHistoryForApi(historyForPrompting, 'openai');
        const geminiHistoryForApi = convertHistoryForApi(historyForPrompting, 'gemini');

        let gptResult = '';
        let geminiResult = '';
        let gptFailed = false;
        let geminiFailed = false;

        const gptStreamPromise = (async () => {
            try {
                const stream = await openai.chat.completions.create({ model: config.gpt, messages: [...gptHistoryForApi, { role: 'user', content: prompt }], stream: true });
                for await (const chunk of stream) {
                    const text = chunk.choices[0]?.delta?.content || "";
                    gptResult += text;
                    sendEvent('modelA-chunk', { text });
                }
            } catch (e) { gptFailed = true; sendEvent('modelA-chunk', { text: `\n\n--- ERROR: OpenAI API call failed. ---` }); }
        })();

        const geminiStreamPromise = (async () => {
             try {
                const model = genAI.getGenerativeModel({ model: config.gemini });
                const chat = model.startChat({ history: geminiHistoryForApi });
                const result = await chat.sendMessageStream(prompt);
                for await (const chunk of result.stream) {
                    const text = chunk.text();
                    geminiResult += text;
                    sendEvent('modelB-chunk', { text });
                }
            } catch (e) { geminiFailed = true; sendEvent('modelB-chunk', { text: `\n\n--- ERROR: Google AI API call failed. ---` }); }
        })();
        
        await Promise.all([gptStreamPromise, geminiStreamPromise]);

        // Step 3: Synthesis
        let synthesisPrompt;
        let fallbackLog = '';
        if (gptFailed && geminiFailed) { /* ... */ return res.end(); }
        else if (gptFailed) { synthesisPrompt = getRefinementPrompt(prompt, geminiResult, constraints, historyForPrompting); fallbackLog = "Model A failed. Refining Model B's response."; }
        else if (geminiFailed) { synthesisPrompt = getRefinementPrompt(prompt, gptResult, constraints, historyForPrompting); fallbackLog = "Model B failed. Refining Model A's response."; }
        else { synthesisPrompt = getSynthesisPrompt(prompt, gptResult, geminiResult, constraints, historyForPrompting); }
        
        sendEvent('fallback-log', { log: fallbackLog });
        sendEvent('status', { message: "Synthesizing final response..." });

        // Step 4: Synthesis Stream
        const synthesizerModel = config.synthesizer;
        try {
            const synthHistoryForApi = convertHistoryForApi(historyForPrompting, synthesizerModel.startsWith('gemini') ? 'gemini' : 'openai');
            if (synthesizerModel.startsWith('gemini')) {
                const model = genAI.getGenerativeModel({ model: synthesizerModel });
                const chat = model.startChat({ history: synthHistoryForApi });
                const result = await chat.sendMessageStream(synthesisPrompt);
                for await (const chunk of result.stream) { sendEvent('synthesis-chunk', { text: chunk.text() }); }
            } else {
                const stream = await openai.chat.completions.create({ model: synthesizerModel, messages: [...synthHistoryForApi, { role: 'user', content: synthesisPrompt }], stream: true });
                for await (const chunk of stream) { sendEvent('synthesis-chunk', { text: chunk.choices[0]?.delta?.content || "" }); }
            }
        } catch (e) {
            fallbackLog += " Synthesis step failed. Displaying best available response.";
            sendEvent('fallback-log', { log: fallbackLog });
            const fallbackResponse = !gptFailed ? gptResult : geminiResult;
            sendEvent('synthesis-chunk', { text: `\n\n--- SYNTHESIS FAILED ---\nDisplaying best available base response:\n\n${fallbackResponse}` });
        }

        sendEvent('done', { message: "Stream complete." });
        res.end();
    } catch (error) {
        sendEvent('error', { message: `An unexpected error occurred: ${error.message}` });
        res.end();
    }
});

app.listen(port, () => { console.log(`Server running at http://localhost:${port}`); });

