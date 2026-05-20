export function getSearchFilesDisplayQuery(args: Record<string, unknown>): string {
    if (typeof args.query === 'string') {
        return args.query;
    }

    if (typeof args.pattern === 'string') {
        return args.pattern;
    }

    return '';
}
