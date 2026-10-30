import { Shell } from "@/components/shell";
import { Demo } from "@/components/bilateral-denoising/demo";
import { Results } from "@/components/bilateral-denoising/results";

export const metadata = { title: "bilateral image denoising / nlp & cv lab" };

export default function Page() {
  return (
    <Shell slug="bilateral-denoising">
      <Demo />
      <Results />
      <section className="mt-16 max-w-2xl space-y-3 text-sm leading-relaxed">
        <h2 className="font-serif text-3xl tracking-tight">how it works</h2>
        <p>
          a bilateral filter replaces each pixel by a weighted average of its neighbours. the weight falls off with
          distance (σ_s, in pixels) and with intensity difference (σ_r), so flat regions get smoothed while edges,
          where neighbours differ a lot, are mostly left alone. the catch is that one global σ_s, σ_r has to fit the
          whole image and the noise level.
        </p>
        <p>
          the manual and auto modes run the classic filter as a fragment shader, so it is real time even on the
          webcam. auto mode estimates the noise level from the image itself and looks up σ_s, σ_r from a grid search on
          held-out BSD images. the neural mode keeps the bilateral structure (the output is still a normalised average
          of noisy neighbours) but lets a tiny unet choose σ_s, σ_r and the range guide for every pixel. turn on the
          σ_r overlay to see it: small σ_r on edges and texture, large in flat areas.
        </p>
        <p className="text-mute">
          everything runs locally in your browser, images are never uploaded. the model is 460 kB and loads from the
          hugging face hub.
        </p>
      </section>
    </Shell>
  );
}
