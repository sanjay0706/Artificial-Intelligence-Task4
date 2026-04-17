import { GoogleGenAI } from "@google/genai";

export async function analyzeDetectionLog(log: any[], analysisType: string = 'summary', customPrompt?: string) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not set in environment variables.");
  }

  const ai = new GoogleGenAI({ apiKey });
  
  let typeInstruction = "Provide a brief, professional summary of the activity observed, noting any patterns or potential concerns.";
  
  if (analysisType === 'anomalies') {
    typeInstruction = "Focus specifically on identifying any anomalies, suspicious patterns, or unusual behaviors in the detection log.";
  } else if (analysisType === 'motion') {
    typeInstruction = "Analyze the frequency and timing of motion events. Identify peak activity periods and movement trends.";
  } else if (analysisType === 'custom' && customPrompt) {
    typeInstruction = customPrompt;
  }

  const prompt = `
    Analyze the following object detection log from a security/monitoring system.
    
    Instruction: ${typeInstruction}
    
    Log Data:
    ${JSON.stringify(log, null, 2)}
    
    Response (max 4 sentences):
  `;

  const response = await ai.models.generateContent({
    model: "gemini-3-flash-preview",
    contents: prompt,
  });

  return response.text || "No analysis generated.";
}
