import { notFound } from "next/navigation";
import { projs, bySlug } from "@/lib/projects";
import { Shell } from "@/components/shell";

// fallback for tasks without a demo yet. real demos live in app/p/<slug>/ and take priority.
export function generateStaticParams() {
  return projs.map((p) => ({ slug: p.slug }));
}

export default async function Page({ params }: PageProps<"/p/[slug]">) {
  const { slug } = await params;
  if (!bySlug(slug)) notFound();
  return (
    <Shell slug={slug}>
      <p className="rounded-lg border border-dashed border-line p-8 text-center text-mute">demo coming soon.</p>
    </Shell>
  );
}
