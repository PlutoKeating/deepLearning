// Wiki content is imported as raw markdown strings
// In production, these could be fetched from an API
// For now, we embed the content directly

const wikiModules = import.meta.glob('/references/wiki/*.md', { query: '?raw', import: 'default' });

export async function loadWikiChapter(filename: string): Promise<string> {
  const key = `/references/wiki/${filename}`;
  const loader = wikiModules[key];
  if (!loader) {
    return `# Chapter Not Found\n\nThe requested chapter "${filename}" could not be loaded.`;
  }
  try {
    const content = await loader() as string;
    return content;
  } catch {
    return `# Error Loading Chapter\n\nFailed to load "${filename}".`;
  }
}

export function getAvailableChapters(): string[] {
  return Object.keys(wikiModules).map(k => k.replace('/references/wiki/', ''));
}
