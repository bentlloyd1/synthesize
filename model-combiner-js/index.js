import express from 'express';
import OpenAI from 'openai';
import { GoogleGenerativeAI } from '@google/generative-ai';
import dotenv from 'dotenv';
import path from 'path';
import { performance } from 'perf_hooks';

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

// --- META-PROMPTS ---
function getSynthesisPrompt(prompt, gptResponse, geminiResponse) {
    return `Your persona is a helpful and pragmatic expert. Your task is to synthesize two AI-generated responses into a single, superior answer. Original User Prompt: "${prompt}"\n\n---\nResponse A:\n"${gptResponse}"\n\n---\nResponse B:\n"${geminiResponse}"\n\n---\nInstructions:\n1. Assess Helpfulness: Prioritize information from the more helpful response.\n2. Extract Core Information: Identify key facts.\n3. Synthesize: Create a single, cohesive answer.\n4. Final Output: Be direct. Do not mention the synthesis process.\n\nSynthesized and Superior Response:`;
}

function getRefinementPrompt(prompt, singleResponse) {
    return `Your persona is a helpful and pragmatic expert. Your task is to review and improve the following AI-generated response. The goal is to make it clearer, more comprehensive, and more direct. Original User Prompt: "${prompt}"\n\n---\nAI Response to Refine:\n"${singleResponse}"\n\n---\nInstructions:\n1. Analyze the response for clarity, accuracy, and completeness.\n2. Rewrite it to be a superior version. Ensure it directly answers the user's prompt.\n3. Do not mention the refinement process.\n\nRefined and Superior Response:`;
}

function getClassificationPrompt(prompt) {
    return `You are an expert at analyzing user prompts. Your task is to classify the user's intent as either "FACTUAL" or "CREATIVE".\n\n- "FACTUAL" intent involves requests for explanations, code, technical information, summaries, or structured answers.\n- "CREATIVE" intent involves requests for stories, brainstorming, poetry, or other open-ended generative tasks.\n\nFirst, provide a brief, step-by-step reasoning for your decision in one or two sentences.\nSecond, on a new line, provide the final classification as a single word: FACTUAL or CREATIVE.\n\nUser Prompt: "${prompt}"\n\nReasoning:\nClassification:`;
}

// --- API CALLS ---
async function getGptResponse(prompt, modelName) {
    try {
        const response = await openai.chat.completions.create({ model: modelName, messages: [{ role: "user", content: prompt }] });
        return response.choices[0].message.content;
    } catch (error) {
        console.error(`Error from OpenAI (${modelName}):`, error);
        return `ERROR: OpenAI API call failed for model ${modelName}.`;
    }
}

async function getGeminiResponse(prompt, modelName) {
    try {
        const model = genAI.getGenerativeModel({ model: modelName });
        const result = await model.generateContent(prompt);
        return result.response.text();
    } catch (error) {
        console.error(`Error from Google AI (${modelName}):`, error);
        return `ERROR: Google AI API call failed for model ${modelName}.`;
    }
}

// --- THE /combine ENDPOINT ---
app.post('/combine', async (req, res) => {
    const { prompt } = req.body;
    if (!prompt) {
        return res.status(400).json({ error: "Prompt is required." });
    }

    try {
        // Step 1: Classify intent
        const classificationPrompt = getClassificationPrompt(prompt);
        const classificationResponse = await getGptResponse(classificationPrompt, CLASSIFIER_MODEL);
        const classifierLines = classificationResponse.trim().split('\n');
        const finalClassification = classifierLines.pop().trim().toUpperCase();
        const reasoning = classifierLines.join(' ').replace('Reasoning:', '').trim();
        const intent = finalClassification.includes("CREATIVE") ? "CREATIVE" : "FACTUAL";

        // Step 2: Select pipeline
        const pipeline = DYNAMIC_PIPELINES[intent];
        const config = pipeline.config;

        // Step 3: Base Generation (in parallel)
        const [gptResult, geminiResult] = await Promise.all([
            getGptResponse(prompt, config.gpt),
            getGeminiResponse(prompt, config.gemini)
        ]);

        // Step 4: Handle fallbacks and prepare for synthesis
        const gptFailed = gptResult.includes('ERROR');
        const geminiFailed = geminiResult.includes('ERROR');
        let synthesisPrompt;
        let fallbackLog = '';

        if (gptFailed && geminiFailed) {
            return res.json({ finalResponse: "FATAL: Both primary models failed. Please try again.", gptResult, geminiResult, pipelineName: pipeline.name, fallbackLog: "Both base models failed.", classifierReasoning: reasoning });
        } else if (gptFailed) {
            synthesisPrompt = getRefinementPrompt(prompt, geminiResult);
            fallbackLog = "A primary model was unavailable. Refining the best available response.";
        } else if (geminiFailed) {
            synthesisPrompt = getRefinementPrompt(prompt, gptResult);
            fallbackLog = "A primary model was unavailable. Refining the best available response.";
        } else {
            synthesisPrompt = getSynthesisPrompt(prompt, gptResult, geminiResult);
        }

        // Step 5: Run Synthesis
        const synthesizerModel = config.synthesizer;
        const synthFn = synthesizerModel.startsWith('gemini') ? getGeminiResponse : getGptResponse;
        let finalResponse = await synthFn(synthesisPrompt, synthesizerModel);

        if (finalResponse.includes('ERROR')) {
            finalResponse = !gptFailed ? gptResult : geminiResult;
            fallbackLog += (fallbackLog ? " " : "") + "The final synthesis step failed. Displaying the best available response.";
        }

        res.json({ finalResponse, gptResult, geminiResult, pipelineName: pipeline.name, fallbackLog, classifierReasoning: reasoning });

    } catch (error) {
        console.error("Error in /combine endpoint:", error);
        res.status(500).json({ error: "An unexpected error occurred." });
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});