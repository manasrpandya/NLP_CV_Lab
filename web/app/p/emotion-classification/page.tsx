import { Shell } from "@/components/shell";
import { Demo } from "@/components/emotion-classification/demo";
import { Results } from "@/components/emotion-classification/results";

export const metadata = { title: "emotion classification | nlp & cv lab" };

export default function Page() {
  return (
    <Shell slug="emotion-classification">
      <Demo />
      <Results />
      <section className="mt-14 max-w-2xl space-y-3 text-sm text-mute">
        <h2 className="font-serif text-3xl text-ink">how it works</h2>
        <p>
          a roberta-large teacher is fine-tuned on GoEmotions (multi-label, 27 emotions + neutral). its soft labels
          are distilled into distilroberta-base, which is exported to ONNX, quantized to int8 and runs here in a
          web worker with transformers.js (webgpu when available, otherwise wasm). nothing leaves your browser.
        </p>
        <p>
          probabilities are temperature-scaled, and every label has its own decision threshold picked on the
          validation set, because rare labels like grief or relief need a much lower cutoff than 0.5. the word
          shading is occlusion: the text is re-run once per word with that word deleted, all in one batch.
        </p>
      </section>
    </Shell>
  );
}
