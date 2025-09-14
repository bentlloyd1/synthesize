import fs from 'fs';
import OpenAI from 'openai';
import { GoogleGenerativeAI } from '@google/generative-ai';
import dotenv from 'dotenv';
import ora from 'ora';
import { performance } from 'perf_hooks';

dotenv.config();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// --- 1. CONFIGURATION ---

// UPDATED: Prompts are now objects to include optional constraints
const PROMPTS_TO_TEST = [
    { 
        prompt: "asdf", 
        constraints: "" 
    },
    { 
        prompt: "Explain the concept of 'technical debt'", 
        constraints: "without using any analogies" 
    },
    { 
        prompt: "Write a short, suspenseful story that starts with the line: 'The old clock chimed thirteen times.'", 
        constraints: "The story must include a character named Elias." 
    },
    { 
        prompt: "Generate a Python function that takes a URL and returns the top 5 most common words on the page.", 
        constraints: "The function should not use any external libraries like requests or BeautifulSoup."
    }
];

const OUTPUT_FILENAME = "synthesis_log_standalone.txt";
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

const PRICING = {
    'gpt-4o': { input: 0.005, output: 0.015 },
    'gemini-1.5-pro-latest': { input: 0.0035, output: 0.0105 },
    'gpt-5': { input: 0.00125, output: 0.01 },
    'gemini-2.5-pro': { input: 0.00125, output: 0.01 }
};

// --- 2. META-PROMPTS ---

// UPDATED: Prompts now accept and apply constraints
function getSynthesisPrompt(prompt, gptResponse, geminiResponse, constraints = '') {
    const constraintInstruction = constraints ? `\n\nCRITICAL INSTRUCTION: You must adhere to the following user-defined constraint when generating the final response:\n- ${constraints}\n` : '';
    return `Your persona is a helpful and pragmatic expert. Your task is to synthesize two AI-generated responses into a single, superior answer. Original User Prompt: "${prompt}"\n\n---\nResponse A:\n"${gptResponse}"\n\n---\nResponse B:\n"${geminiResponse}"\n\n---${constraintInstruction}\n\nInstructions:\n1. Synthesize the responses into a single, cohesive answer that strictly follows the user's constraints if provided.\n2. Be direct. Do not mention the synthesis process.\n\nSynthesized and Superior Response:`;
}

function getRefinementPrompt(prompt, singleResponse, constraints = '') {
    const constraintInstruction = constraints ? `\n\nCRITICAL INSTRUCTION: You must adhere to the following user-defined constraint when generating the final response:\n- ${constraints}\n` : '';
    return `Your persona is a helpful and pragmatic expert. Your task is to review and improve the following AI-generated response. Original User Prompt: "${prompt}"\n\n---\nAI Response to Refine:\n"${singleResponse}"\n\n---${constraintInstruction}\n\nInstructions:\n1. Rewrite the response to be a superior version that strictly follows the user's constraints if provided.\n2. Do not mention the refinement process.\n\nRefined and Superior Response:`;
}

function getClassificationPrompt(prompt) {
    return `You are an expert at analyzing user prompts. Your task is to classify the user's intent as either "FACTUAL" or "CREATIVE".\n\n- "FACTUAL" intent involves requests for explanations, code, technical information, summaries, or structured answers.\n- "CREATIVE" intent involves requests for stories, brainstorming, poetry, or other open-ended generative tasks.\n\nFirst, provide a brief, step-by-step reasoning for your decision in one or two sentences.\nSecond, on a new line, provide the final classification as a single word: FACTUAL or CREATIVE.\n\nUser Prompt: "${prompt}"\n\nReasoning:\nClassification:`;
}


// --- 3. API CALLS & COST TRACKING (Unchanged from previous script) ---
let totalCost = 0;
async function runStep(apiCallFn) { const start = performance.now(); const result = await apiCallFn(); const duration = performance.now() - start; totalCost += result.cost; return { ...result, duration }; }
async function getGptResponse(prompt, modelName) { try { const response = await openai.chat.completions.create({ model: modelName, messages: [{ role: "user", content: prompt }] }); const usage = response.usage; const cost = ((usage.prompt_tokens / 1000) * PRICING[modelName].input) + ((usage.completion_tokens / 1000) * PRICING[modelName].output); return { content: response.choices[0].message.content, cost }; } catch (error) { return { content: `ERROR from OpenAI: ${error.message}`, cost: 0 }; } }
async function getGeminiResponse(prompt, modelName) { try { const model = genAI.getGenerativeModel({ model: modelName }); let pT = 0, cT = 0; try { pT = (await model.countTokens(prompt)).totalTokens; } catch (e) { pT = Math.ceil(prompt.length / 4); } const result = await model.generateContent(prompt); const rT = result.response.text(); try { cT = (await model.countTokens(rT)).totalTokens; } catch (e) { cT = Math.ceil(rT.length / 4); } const cost = ((pT / 1000) * PRICING[modelName].input) + ((cT / 1000) * PRICING[modelName].output); return { content: rT, cost }; } catch (error) { return { content: `ERROR from Google AI: ${error.message}`, cost: 0 }; } }


// --- 4. MAIN SCRIPT EXECUTION ---
async function main() {
    const spinner = ora('Setting up resilient tests...').start();
    fs.writeFileSync(OUTPUT_FILENAME, '', 'utf-8');

    const baseResponseCache = {};
    const allTestPromises = PROMPTS_TO_TEST.map(testCase => runDynamicTest({ ...testCase, cache: baseResponseCache }));
    
    spinner.text = `Running ${allTestPromises.length} resilient tests in parallel...`;
    const results = await Promise.all(allTestPromises);
    
    results.sort((a, b) => PROMPTS_TO_TEST.findIndex(p => p.prompt === a.prompt) - PROMPTS_TO_TEST.findIndex(p => p.prompt === b.prompt));

    spinner.text = `Writing ${results.length} results to ${OUTPUT_FILENAME}...`;
    fs.writeFileSync(OUTPUT_FILENAME, results.map(r => r.txtEntry).join(''), 'utf-8');

    spinner.succeed("Success! All tests processed.");
    
    console.log("\n--- Performance Summary ---");
    results.forEach(r => {
        console.log(`[Prompt: "${r.prompt.substring(0, 20)}..."]`);
        console.log(`  - Classifier Reason: ${r.classifierReasoning.replace(/\n/g, ' ')}`);
        console.log(`  -> Routed to: ${r.pipelineName}`);
        if (r.fallbackLog) console.log(`  - ${r.fallbackLog}`);
        console.log(`  - Classifier (${CLASSIFIER_MODEL}): ${(r.classifierDuration / 1000).toFixed(2)}s, $${r.classifierCost.toFixed(5)}`);
        console.log(`  - GPT Base (${r.gptModel}): ${(r.gptBaseDuration / 1000).toFixed(2)}s, $${r.gptBaseCost.toFixed(5)}`);
        console.log(`  - Gemini Base (${r.geminiModel}): ${(r.geminiBaseDuration / 1000).toFixed(2)}s, $${r.geminiBaseCost.toFixed(5)}`);
        console.log(`  - Synthesis (${r.synthModel}): ${(r.synthDuration / 1000).toFixed(2)}s, $${r.synthCost.toFixed(5)}`);
    });

    console.log("\n-------------------------------------------------");
    console.log(`- Results saved to '${OUTPUT_FILENAME}'`);
    console.log(`💰 Total estimated cost for this run: $${totalCost.toFixed(4)}`);
    console.log("-------------------------------------------------");
}

// UPDATED: Now accepts `constraints`
async function runDynamicTest({ prompt, constraints, cache }) {
    // Step 1: Classify intent
    const classificationPrompt = getClassificationPrompt(prompt);
    const classifierRun = await runStep(() => getGptResponse(classificationPrompt, CLASSIFIER_MODEL));
    
    const classifierLines = classifierRun.content.trim().split('\n');
    const finalClassification = classifierLines.pop().trim().toUpperCase();
    const reasoning = classifierLines.join(' ').replace('Reasoning:', '').trim();
    const intent = finalClassification.includes("CREATIVE") ? "CREATIVE" : "FACTUAL";

    const pipeline = DYNAMIC_PIPELINES[intent];
    const config = pipeline.config;
    
    // Step 2: Get base responses
    const gptCacheKey = `${prompt}-${config.gpt}`;
    const geminiCacheKey = `${prompt}-${config.gemini}`;
    let gptRun = cache[gptCacheKey] || await runStep(() => getGptResponse(prompt, config.gpt));
    cache[gptCacheKey] = gptRun;
    let geminiRun = cache[geminiCacheKey] || await runStep(() => getGeminiResponse(prompt, config.gemini));
    cache[geminiCacheKey] = geminiRun;

    // Step 3: Handle fallbacks
    const gptFailed = gptRun.content.includes('ERROR');
    const geminiFailed = geminiRun.content.includes('ERROR');
    let synthesisPrompt;
    let fallbackLog = '';

    if (gptFailed && geminiFailed) {
        // ... (catastrophic failure logic)
        const result = { prompt, constraints, pipelineName: pipeline.name, fallbackLog: "FALLBACK: Both base models failed.", classifierReasoning: reasoning, classifierDuration: classifierRun.duration, classifierCost: classifierRun.cost, gptModel: config.gpt, gptBaseDuration: gptRun.duration, gptBaseCost: gptRun.cost, geminiModel: config.gemini, geminiBaseDuration: geminiRun.duration, geminiBaseCost: geminiRun.cost, synthModel: config.synthesizer, synthDuration: 0, synthCost: 0, gptRun, geminiRun };
        result.txtEntry = createTxtEntry({ ...result, finalResponse: "FATAL: Both base models failed.", config });
        return result;
    } else if (gptFailed) {
        synthesisPrompt = getRefinementPrompt(prompt, geminiRun.content, constraints);
        fallbackLog = "FALLBACK: GPT-5 base failed. Refining Gemini response.";
    } else if (geminiFailed) {
        synthesisPrompt = getRefinementPrompt(prompt, gptRun.content, constraints);
        fallbackLog = "FALLBACK: Gemini base failed. Refining GPT-5 response.";
    } else {
        synthesisPrompt = getSynthesisPrompt(prompt, gptRun.content, geminiRun.content, constraints);
    }

    // Step 4: Run synthesis
    const synthesizerModel = config.synthesizer;
    const synthFn = synthesizerModel.startsWith('gemini') ? getGeminiResponse : getGptResponse;
    const synthRun = await runStep(() => synthFn(synthesisPrompt, synthesizerModel));

    // Step 5: Handle synthesizer failure
    let finalResponse = synthRun.content;
    if (finalResponse.includes('ERROR')) {
        finalResponse = !gptFailed ? gptRun.content : geminiRun.content;
        fallbackLog += (fallbackLog ? " " : "") + `FALLBACK: Synthesis step failed; returning ${!gptFailed ? 'GPT-5' : 'Gemini'} base.`;
    }
    
    // Step 6: Assemble results
    const resultData = { prompt, constraints, pipelineName: pipeline.name, fallbackLog, classifierReasoning: reasoning, classifierDuration: classifierRun.duration, classifierCost: classifierRun.cost, gptModel: config.gpt, gptBaseDuration: gptRun.duration, gptBaseCost: gptRun.cost, geminiModel: config.gemini, geminiBaseDuration: geminiRun.duration, geminiBaseCost: geminiRun.cost, synthModel: synthesizerModel, synthDuration: synthRun.duration, synthCost: synthRun.cost, finalResponse, config, gptRun, geminiRun };
    resultData.txtEntry = createTxtEntry(resultData);
    return resultData;
}

// UPDATED: Now includes constraints in the log output
function createTxtEntry({ prompt, constraints, pipelineName, fallbackLog, config, gptRun, geminiRun, finalResponse }) {
    const finalResponseTitle = fallbackLog.includes('Synthesis step failed') ? 'FALLBACK RESPONSE' : 'FINAL RESPONSE';
    
    return `\n\n######################################################################
## DYNAMICALLY ROUTED TO: ${pipelineName}
${fallbackLog ? `## FALLBACK TRIGGERED: ${fallbackLog}\n` : ''}######################################################################
======================================================================
== PROMPT: ${prompt}
${constraints ? `== CONSTRAINTS: ${constraints}\n` : ''}======================================================================
--- MODEL A (${config.gpt}) RESPONSE ---
${gptRun.content}
----------------------------------------------------------------------
--- MODEL B (${config.gemini}) RESPONSE ---
${geminiRun.content}
----------------------------------------------------------------------
--- ${finalResponseTitle} (using ${config.synthesizer}) ---
${finalResponse}\n`;
}

main().catch(console.error);

