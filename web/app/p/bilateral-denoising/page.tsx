import { Shell } from "@/components/shell";
import { Demo } from "@/components/bilateral-denoising/demo";
import { Results } from "@/components/bilateral-denoising/results";
import { More } from "@/components/bilateral-denoising/tip";

export const metadata = { title: "bilateral image denoising / nlp & cv lab" };

const report = "https://github.com/manasrpandya/NLP_CV_Lab/blob/main/projects/07-bilateral-denoising/report/report.pdf";
const hub = "https://huggingface.co/manasrpandya/bilateral-denoise-nbf";

export default function Page() {
  return (
    <Shell slug="bilateral-denoising">
      <p className="-mt-4 mb-8 max-w-2xl text-sm leading-relaxed">
        grainy photos are noisy photos. this page adds grain to a picture and then removes it again, with three
        versions of the same classic trick. everything runs in your browser, nothing is uploaded.
      </p>
      <Demo />
      <Results />
      <section className="mt-16 max-w-2xl">
        <h2 className="font-serif text-3xl tracking-tight">details</h2>
        <div className="mt-4 border-t border-line">
          <More title="what is a bilateral filter?">
            <p>
              it replaces each pixel by a weighted average of its neighbours. a neighbour counts less the further away
              it is (σ_s, in pixels) and the more its brightness differs (σ_r). so flat areas get smoothed, while across
              an edge the neighbours differ a lot and are mostly ignored.
            </p>
            <p>
              the catch: one σ_s and σ_r for the whole image has to suit both flat sky and fine texture, and with heavy
              noise the filter can no longer tell noise from edges.
            </p>
          </More>
          <More title="how does auto pick its settings?">
            <p>
              it measures the noise with immerkaer&apos;s method, a small filter that cancels smooth image content so
              mostly noise is left. that number goes into a table of the best settings per noise level, found by a grid
              search on 32 held-out training photos. on artificial noise the estimate is within about 1 grey level.
            </p>
          </More>
          <More title="what does the neural version do?">
            <p>
              a small u-net (93k weights, 460 kB) looks at the noisy photo and outputs σ_s, σ_r and a guide image for
              every pixel. two bilateral passes then use those maps. the result is still an average of nearby noisy
              pixels, so it cannot invent detail. it was trained on 400 photos with random noise levels and never gets
              told how noisy the input is.
            </p>
            <p>
              it runs with onnx runtime in a background thread, on your gpu (webgpu) if possible and on the cpu
              otherwise, in 256 px tiles so it fits in a phone&apos;s memory.
            </p>
          </More>
          <More title="where does it fall short?">
            <p>
              the same network without the bilateral structure scores just as well, so the structure buys
              explainable maps rather than quality. a bigger network (DnCNN) is 0.8 dB better. on real camera noise the
              network barely helps, because it was trained on artificial noise only.
            </p>
          </More>
          <More title="report, code and model">
            <p>
              the full write-up with all experiments is in the{" "}
              <a className="underline" href={report}>
                report
              </a>
              . the model is on the{" "}
              <a className="underline" href={hub}>
                hugging face hub
              </a>
              .
            </p>
          </More>
        </div>
      </section>
    </Shell>
  );
}
