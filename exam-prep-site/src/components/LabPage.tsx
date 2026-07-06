import { useState, useEffect } from 'react';
import MarkdownRenderer from '../components/MarkdownRenderer';

interface LabPageProps {
  name: string;
  file: string;
}

export default function LabPage({ name, file }: LabPageProps) {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/references/wiki/${file}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        setContent(text || `# ${name}\n\nContent pending.`);
        setLoading(false);
      })
      .catch(() => {
        setContent(`# ${name}\n\nLab content is being generated. Please check back soon.`);
        setLoading(false);
      });
  }, [file, name]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400" />
        <span className="ml-3 text-gray-400">Loading lab content...</span>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <MarkdownRenderer content={content} />
    </div>
  );
}
