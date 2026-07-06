import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

interface MarkdownRendererProps {
  content: string;
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="prose-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
        components={{
          pre({ children }) {
            return (
              <pre className="bg-gray-900 border border-gray-700 rounded-lg p-4 my-4 overflow-x-auto text-sm">
                {children}
              </pre>
            );
          },
          code({ className, children, ...props }) {
            const isInline = !className;
            if (isInline) {
              return (
                <code className="bg-gray-800 text-pink-300 px-1.5 py-0.5 rounded text-sm" {...props}>
                  {children}
                </code>
              );
            }
            return (
              <code className={`${className || ''} text-gray-200`} {...props}>
                {children}
              </code>
            );
          },
          table({ children }) {
            return (
              <div className="overflow-x-auto my-4">
                <table className="w-full border-collapse">{children}</table>
              </div>
            );
          },
          th({ children }) {
            return (
              <th className="bg-gray-800 text-left px-4 py-2 border border-gray-600 font-semibold text-white">
                {children}
              </th>
            );
          },
          td({ children }) {
            return (
              <td className="px-4 py-2 border border-gray-700">{children}</td>
            );
          },
          blockquote({ children }) {
            return (
              <blockquote className="border-l-4 border-blue-500 pl-4 my-4 text-gray-400 italic">
                {children}
              </blockquote>
            );
          },
          a({ href, children }) {
            return (
              <a href={href} className="text-blue-400 underline hover:text-blue-300" target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            );
          },
          img({ src, alt }) {
            return <img src={src} alt={alt || ''} className="max-w-full rounded-lg my-4" loading="lazy" />;
          },
          hr() {
            return <hr className="border-gray-700 my-8" />;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
