import { useState, useEffect } from 'react';
import MarkdownRenderer from '../components/MarkdownRenderer';
import type { Chapter } from '../data/chapters';

interface ChapterPageProps {
  chapter: Chapter;
}

export default function ChapterPage({ chapter }: ChapterPageProps) {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    setError('');
    fetch(`/references/wiki/${chapter.file}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        if (!text || text.trim().length < 10) throw new Error('Empty content');
        setContent(text);
        setLoading(false);
      })
      .catch(() => {
        setError(`第 ${chapter.id} 章内容正在努力加载或同步中，请稍后刷新重试... Chapter content is syncing.`);
        setLoading(false);
      });
  }, [chapter.file]);

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-sm font-mono text-purple-400 bg-purple-400/10 px-2 py-0.5 rounded">
            第 {chapter.id} 章 Chapter {chapter.id}
          </span>
          <h1 className="text-4xl font-bold text-white">{chapter.title}</h1>
        </div>
        <p className="text-gray-400 text-lg">{chapter.subtitle}</p>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-20">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400" />
          <span className="ml-3 text-gray-400">正在加载内容... Loading content...</span>
        </div>
      )}

      {error && (
        <div className="bg-yellow-900/30 border border-yellow-700/50 rounded-lg p-6 text-yellow-200">
          <p className="font-semibold mb-2">⏳ 正在同步内容 Content Syncing</p>
          <p>{error}</p>
        </div>
      )}

      {!loading && !error && (
        <MarkdownRenderer content={content} />
      )}

      {!loading && !error && content && (
        <div className="mt-12 pt-6 border-t border-gray-800 text-gray-500 text-sm">
          <p>📖 内容源自课程讲义、学术经典论文与标准深度学习参考资料。 Sourced from course lecture slides and academic references.</p>
          <p className="mt-1">🟢 来自资料 · 🟡 AI补充 · ⚠️ AI生成答案</p>
        </div>
      )}
    </div>
  );
}
