/**
 * Query Flow
 * Semantic query against the Wiki knowledge base
 */

import { App, TFile } from 'obsidian';
import type { LLMWikiSettings, OllamaMessage, ToolContext, QueryResult } from '../types';
import { getLLMClient } from '../llm/client';
import { executeTool, getOllamaTools } from '../tools';
import { buildRegexFilteredIndex } from './indexContext';

const SYSTEM_PROMPT = `You are a knowledge base query assistant. Your task is to answer user questions based STRICTLY on the Wiki knowledge base content.

## CRITICAL CONSTRAINT: Knowledge Base Reliance
**STRICTLY PROHIBITED**: You must NOT use any external knowledge, prior knowledge, or information not present in the Wiki.

**YOU MUST**:
- Answer ONLY based on content found in Wiki pages
- Cite every piece of information using [[page-name]] format
- State clearly if the Wiki does not contain relevant information

**FORBIDDEN**:
- Using your own knowledge to supplement answers
- Making inferences beyond what's explicitly stated in Wiki pages
- Answering questions when the Wiki lacks relevant information
- Providing information without proper [[page-name]] citations

## Workflow
1. First read the Wiki index (index.md) to understand the knowledge base structure
2. Locate candidate Wiki pages based on the question
3. For each candidate, read frontmatter first (Read_Property), then summary (Read_Summary)
4. Only when relevance is high, read full content (read_file) or the needed section (Read_Part)
5. Synthesize answers ONLY from the content you have read
6. Mark citation sources in the answer using [[page-name]] format

## Retrieval Strategy (MUST FOLLOW)
- Prioritize cheap reads first: Read_Property -> Read_Summary -> read_file/Read_Part
- Use read_file only for high-match pages
- High match means the question intent clearly aligns with title/tags/related and summary keywords
- If relevance remains low after summary, skip full-text read and continue with other candidates

## Citation Format
- Every factual statement must be followed by its source: [[page-name]]
- Multiple sources: [[page-1]] [[page-2]]
- Example: "Python is a programming language. [[Python]]"

## Answer Guidelines
- Answers should be accurate and concise
- MUST cite ALL information sources using [[page-name]] format
- If Wiki lacks relevant information, respond: "The knowledge base does not contain information about [topic]."
- Clearly state if information is incomplete or uncertain
- Use Markdown format
- Use [[wikilinks]] syntax for related concepts

## Available Tools
- read_file: Read file contents
- Read_Property: Read only one frontmatter property
- Read_Summary: Read only the Summary section
- Read_Part: Read only one named section
- search_files: Search file contents
- list_files: List directory files`;

/**
 * Query the Wiki knowledge base
 */
export async function queryWiki(
    app: App,
    settings: LLMWikiSettings,
    question: string,
    onChunk?: (text: string) => void
): Promise<QueryResult> {
    const client = getLLMClient(settings);
    const context: ToolContext = {
        vault: app.vault,
        app,
        settings,
    };

    try {
        // Read Wiki index first
        const indexPath = `${settings.wikiPath}/index.md`;
        const indexFile = app.vault.getAbstractFileByPath(indexPath);
        let indexContent = '';
        
        if (indexFile instanceof TFile) {
            indexContent = await app.vault.read(indexFile);
        }

        const filteredIndexContent = buildRegexFilteredIndex(indexContent, question);

        // Build initial message
        const messages: OllamaMessage[] = [
            {
                role: 'user',
                content: `Please answer the following question:

## Question
${question}

## Regex-Matched Wiki Index Blocks
\`\`\`
${filteredIndexContent}
\`\`\`

The index excerpt above was filtered from index.md using regex matches derived from the question. If it is insufficient, read index.md or other Wiki pages with tools before answering.`,
            },
        ];

        // Run agentic loop
        const tools = getOllamaTools();
        let response = (await client.chat({ messages, tools, systemPrompt: SYSTEM_PROMPT })).message;
        let iterations = 0;
        const maxIterations = 5;
        const sources: string[] = [];

        while (iterations < maxIterations) {
            iterations++;

            if (response.toolCalls && response.toolCalls.length > 0) {
                for (const toolCall of response.toolCalls) {
                    const result = await executeTool(
                        toolCall.function.name,
                        toolCall.function.arguments,
                        context
                    );

                    // Track which pages were read
                    if (
                        toolCall.function.name === 'read_file' ||
                        toolCall.function.name === 'Read_Property' ||
                        toolCall.function.name === 'Read_Summary' ||
                        toolCall.function.name === 'Read_Part'
                    ) {
                        const path = toolCall.function.arguments.path as string;
                        if (path.startsWith(settings.wikiPath) && !sources.includes(path)) {
                            sources.push(path);
                        }
                    }

                    messages.push({
                        role: 'assistant',
                        content: '',
                        toolCalls: response.toolCalls,
                    });
                    messages.push({
                        role: 'tool',
                        content: JSON.stringify(result),
                        toolCallId: toolCall.id,
                    });
                }

                response = (await client.chat({ messages, tools, systemPrompt: SYSTEM_PROMPT })).message;
            } else {
                break;
            }
        }

        // Stream the final response if callback provided
        if (onChunk && response.content) {
            onChunk(response.content);
        }

        // Extract page titles from source paths
        const sourceTitles = sources.map((path) => {
            const match = path.match(/([^/]+)\.md$/);
            return match ? match[1] : path;
        });

        return {
            answer: response.content || 'Unable to generate answer',
            sources: sourceTitles,
            confidence: sources.length > 0 ? 0.8 : 0.3,
        };
    } catch (error) {
        return {
            answer: `Query failed: ${error}`,
            sources: [],
            confidence: 0,
        };
    }
}

/**
 * Chat with the Wiki in streaming mode
 */
export async function chatWiki(
    app: App,
    settings: LLMWikiSettings,
    messages: OllamaMessage[],
    onChunk: (text: string) => void,
    contextPrompt?: string
): Promise<string> {
    const client = getLLMClient(settings);
    const context: ToolContext = {
        vault: app.vault,
        app,
        settings,
    };

    try {
        const tools = getOllamaTools();
        const systemPrompt = contextPrompt ? `${contextPrompt}\n\n${SYSTEM_PROMPT}` : SYSTEM_PROMPT;
        let response = (await client.chatStream({
            messages,
            onChunk,
            tools,
            systemPrompt,
        })).message;
        let iterations = 0;
        const maxIterations = 5;

        while (iterations < maxIterations) {
            iterations++;

            if (response.toolCalls && response.toolCalls.length > 0) {
                for (const toolCall of response.toolCalls) {
                    const result = await executeTool(
                        toolCall.function.name,
                        toolCall.function.arguments,
                        context
                    );

                    messages.push({
                        role: 'assistant',
                        content: '',
                        toolCalls: response.toolCalls,
                    });
                    messages.push({
                        role: 'tool',
                        content: JSON.stringify(result),
                        toolCallId: toolCall.id,
                    });
                }

                response = (await client.chatStream({
                    messages,
                    onChunk,
                    tools,
                    systemPrompt,
                })).message;
            } else {
                break;
            }
        }

        return response.content;
    } catch (error) {
        return `Conversation failed: ${error}`;
    }
}
