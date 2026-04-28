const QUESTION_STOP_WORDS = new Set([
    'a', 'an', 'and', 'are', 'at', 'be', 'by', 'for', 'from', 'how', 'in', 'is', 'it', 'of', 'on',
    'or', 'that', 'the', 'this', 'to', 'was', 'what', 'when', 'where', 'which', 'who', 'why', 'with',
    '请', '关于', '什么', '怎么', '如何', '哪些', '是否', '一下', '一个', '一些', '这个', '那个', '以及', '还有'
]);

function escapeRegExp(text: string): string {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function extractQuestionKeywords(question: string): string[] {
    const wikiLinks = Array.from(question.matchAll(/\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g), (match) => match[1]);
    const textTokens = question.match(/[A-Za-z0-9_/-]{2,}|[\u4e00-\u9fff]{2,}/g) || [];
    const tokens = [...wikiLinks, ...textTokens]
        .map((token) => token.trim().toLowerCase())
        .filter((token) => token.length >= 2 && !QUESTION_STOP_WORDS.has(token));

    return Array.from(new Set(tokens)).sort((left, right) => right.length - left.length).slice(0, 8);
}

function splitIndexBlocks(indexContent: string): string[] {
    const normalizedContent = indexContent.replace(/\r\n/g, '\n');
    const headerMatch = normalizedContent.match(/^[\s\S]*?(?=^###\s+)/m);
    const header = headerMatch?.[0].trim();
    const sectionMatches = normalizedContent.match(/^###\s+[\s\S]*?(?=^###\s+|\Z)/gm) || [];
    const blocks = sectionMatches.map((section) => section.trim()).filter(Boolean);

    return header ? [header, ...blocks] : blocks;
}

export function buildRegexFilteredIndex(indexContent: string, question: string, maxBlocks: number = 6): string {
    if (!indexContent.trim()) {
        return '(Wiki is empty)';
    }

    const blocks = splitIndexBlocks(indexContent);
    if (blocks.length === 0) {
        return indexContent;
    }

    const [headerBlock, ...contentBlocks] = blocks;
    const keywords = extractQuestionKeywords(question);
    if (keywords.length === 0) {
        return blocks.slice(0, Math.min(maxBlocks, blocks.length)).join('\n\n');
    }

    const matchedBlocks = contentBlocks
        .map((block) => {
            const score = keywords.reduce((total, keyword) => {
                const regex = new RegExp(escapeRegExp(keyword), 'gi');
                const matches = block.match(regex);
                return total + (matches?.length || 0);
            }, 0);

            return { block, score };
        })
        .filter((item) => item.score > 0)
        .sort((left, right) => right.score - left.score)
        .slice(0, maxBlocks)
        .map((item) => item.block);

    if (matchedBlocks.length === 0) {
        return blocks.slice(0, Math.min(maxBlocks, blocks.length)).join('\n\n');
    }

    return [headerBlock, ...matchedBlocks].filter(Boolean).join('\n\n');
}