import { Shell } from "@/components/shell";
import { Demo } from "@/components/emotion-classification/demo";
import { Results } from "@/components/emotion-classification/results";
import { More } from "@/components/emotion-classification/tip";

export const metadata = { title: "text emotion classification / nlp & cv lab" };

const report =
  "https://github.com/manasrpandya/NLP_CV_Lab/blob/main/projects/06-emotion-classification/report/report.pdf";
const code = "https://github.com/manasrpandya/NLP_CV_Lab/tree/main/projects/06-emotion-classification";
const hub = "https://huggingface.co/manasrpandya/goemotions-distil-onnx";

export default function Page() {
  return (
    <Shell slug="emotion-classification">
      <p className="-mt-4 mb-8 max-w-2xl text-sm leading-relaxed">
        type a message and see which emotions it carries, out of 27 plus neutral, and which words gave them away.
        everything runs in your browser, nothing you type is uploaded.
      </p>
      <Demo />
      <Results />
      <section className="mt-16 max-w-2xl">
        <h2 className="font-serif text-3xl tracking-tight">details</h2>
        <div className="mt-4 border-t border-line">
          <More title="where does the data come from?">
            <p>
              GoEmotions, a set of 58k reddit comments released by google in 2020. each comment was labelled by three
              or more people with any of 27 emotions, or neutral. a comment can have several at once (about 1 in 6
              do). some emotions are very rare: grief has only 77 training examples, neutral has 14,000.
            </p>
          </More>
          <More title="how does the model work?">
            <p>
              first a big model (roberta-large, 355M weights) was trained on the comments, twice with two different
              losses. then a small model (distilroberta, 82M weights) was trained to copy the big models&apos; softened
              answers as well as the human labels. this is called knowledge distillation, and it gave the small model
              about +0.014 macro-f1 over training it on the labels alone.
            </p>
            <p>
              the small model was exported to ONNX and stored in 8 bits per weight (79 mb instead of 313 mb), which
              changes its score by about 0.001. here it runs with transformers.js in a background thread.
            </p>
          </More>
          <More title="why does every emotion have its own cutoff?">
            <p>
              a model is rarely confident about rare emotions, so a 50% cutoff would almost never predict grief or
              relief. the cutoff for each emotion was picked on separate validation comments to get the best f1, after
              smoothing, because with only 13 grief examples the best value jumps around a lot.
            </p>
          </More>
          <More title="where does it fall short?">
            <p>
              sarcasm and irony often fool it (&ldquo;great start&rdquo; reads as positive). rare or subtle emotions like
              realization, relief and grief are still hard, with f1 between 0.2 and 0.45. it learned from reddit, so
              very different text (formal writing, other languages) will be read less reliably. on tweets it got the
              main emotion right 63% of the time.
            </p>
          </More>
          <More title="report, code and model">
            <p>
              the full write-up with all experiments is in the{" "}
              <a className="underline" href={report}>
                report
              </a>
              , the code is on{" "}
              <a className="underline" href={code}>
                github
              </a>{" "}
              and the model on the{" "}
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
