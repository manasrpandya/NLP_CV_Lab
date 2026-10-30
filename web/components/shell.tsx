import Link from "next/link";
import { bySlug, pad } from "@/lib/projects";

// common frame for every demo page: header, back link, footer with repo link
export const repo = "https://github.com/manasrpandya/NLP_CV_Lab";

export function Shell({ slug, children }: { slug: string; children: React.ReactNode }) {
  const p = bySlug(slug);
  return (
    <div className="mx-auto w-full max-w-5xl px-4 py-8 sm:px-6">
      <Link href="/" className="text-sm text-mute hover:text-ink">
        &larr; all tasks
      </Link>
      {p && (
        <header className="mt-6 mb-8">
          <p className="font-mono text-xs text-mute">
            task {pad(p.n)} / {p.area}
          </p>
          <h1 className="mt-1 font-serif text-4xl tracking-tight">{p.title}</h1>
          <p className="mt-2 text-mute">{p.blurb}</p>
        </header>
      )}
      {children}
      <footer className="mt-16 border-t border-line pt-4 text-xs text-mute">
        source:{" "}
        <a className="underline" href={`${repo}/tree/main/projects/${p ? `${pad(p.n)}-${p.slug}` : ""}`}>
          github
        </a>
      </footer>
    </div>
  );
}
