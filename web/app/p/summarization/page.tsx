import { Shell } from "@/components/shell";
import { Demo } from "@/components/summarization/demo";
import { Results } from "@/components/summarization/results";
import { More } from "@/components/ui/tip";

export const metadata = { title: "text summarization / nlp & cv lab" };

const report =
  "https://github.com/manasrpandya/NLP_CV_Lab/blob/main/projects/03-summarization/report/report.pdf";
const code = "https://github.com/manasrpandya/NLP_CV_Lab/tree/main/projects/03-summarization";
const hub = "https://huggingface.co/manasrpandya/cnndm-summ-onnx";

export default function Page() {
  return (
    <Shell slug="summarization">
      <p className="-mt-4 mb-8 max-w-2xl text-sm leading-relaxed">
        shrink a news article to a few sentences, and check whether the summary only says things the article says.
        everything runs in your browser, nothing you paste is uploaded.
      </p>
      <Demo />
      <Results />
      <section className="mt-16 max-w-2xl">
        <h2 className="font-serif text-3xl tracking-tight">details</h2>
        <div className="mt-4 border-t border-line">
          <More title="report, code and model">
            <p>
              the full write-up is in the{" "}
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
