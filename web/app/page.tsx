import Link from "next/link";
import { projs, pad, type st } from "@/lib/projects";
import { repo } from "@/components/shell";

const tag: Record<st, string> = {
  live: "text-acc",
  wip: "text-mute",
  todo: "text-mute/60",
};

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-5xl px-4 py-12 sm:px-6">
      <h1 className="font-serif text-5xl tracking-tight sm:text-6xl">nlp &amp; cv lab</h1>
      <p className="mt-3 max-w-xl text-mute">
        fifteen lab tasks in nlp and computer vision. models are trained in pytorch, exported to onnx and run
        right here in the browser, so nothing gets uploaded anywhere.
      </p>
      <p className="mt-2 text-sm">
        <a className="underline" href={repo}>
          code on github
        </a>
      </p>

      <ul className="mt-10 grid gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-2 lg:grid-cols-3">
        {projs.map((p) => {
          const body = (
            <div className="flex h-full flex-col bg-card p-5">
              <div className="flex items-baseline justify-between font-mono text-xs">
                <span className="text-mute">
                  {pad(p.n)} / {p.area}
                </span>
                <span className={tag[p.status]}>{p.status}</span>
              </div>
              <h2 className="mt-3 text-lg font-medium leading-snug">{p.title}</h2>
              <p className="mt-1 text-sm text-mute">{p.blurb}</p>
            </div>
          );
          return (
            <li key={p.slug}>
              {p.status === "todo" ? (
                body
              ) : (
                <Link href={`/p/${p.slug}`} className="block h-full transition-opacity hover:opacity-80">
                  {body}
                </Link>
              )}
            </li>
          );
        })}
      </ul>
    </main>
  );
}
