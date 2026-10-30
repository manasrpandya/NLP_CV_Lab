import Link from "next/link";
import { projs, pad, type st } from "@/lib/projects";
import { repo } from "@/components/shell";

const tag: Record<st, string> = {
  live: "rounded-sm bg-acc px-1.5 py-0.5 font-medium text-bg",
  wip: "rounded-sm border border-line px-1.5 py-0.5 text-mute",
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
            <div
              className={`flex h-full flex-col p-5 ${p.status === "live" ? "bg-[color-mix(in_srgb,var(--acc)_9%,var(--card))] shadow-[inset_0_3px_0_var(--acc)]" : "bg-card"}`}
            >
              <div className="flex items-center justify-between font-mono text-xs">
                <span className="text-mute">
                  {pad(p.n)} / {p.area}
                </span>
                <span className={`flex items-center gap-1.5 ${tag[p.status]}`}>
                  {p.status === "live" && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />}
                  {p.status}
                </span>
              </div>
              <h2 className="mt-3 text-lg font-medium leading-snug">{p.title}</h2>
              <p className="mt-1 text-sm text-mute">{p.blurb}</p>
              {p.status === "live" && <p className="mt-auto pt-4 text-sm text-acc">try the demo &rarr;</p>}
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
