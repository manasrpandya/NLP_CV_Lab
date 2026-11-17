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
          <More title="how does the ai summary work?">
            <p>
              the model is bart-base, a transformer with an encoder that reads the text and a decoder that writes the
              summary one word piece at a time. it was trained for 20 minutes on 40,000 news articles and their
              human-written highlights. the length buttons work because every training example started with the words
              &ldquo;summarize short / medium / long&rdquo;, picked from how long its human summary was.
            </p>
            <p>
              to fit in a browser the model&apos;s vocabulary was cut from 50,000 to 18,000 word pieces (the ones news
              actually uses) and its weights were stored in 8 bits each, which together take it from 560 mb to about
              137 mb. that costs about one rouge point.
            </p>
          </More>
          <More title="what does the fact check do?">
            <p>
              a second, much smaller model (deberta-v3-xsmall, trained to tell whether one sentence follows from another)
              reads each summary sentence together with every sentence of the text and keeps the best match. a score
              above 0.5 is &ldquo;ok&rdquo;, between 0.1 and 0.5 &ldquo;weak&rdquo;, below that &ldquo;not found in
              text&rdquo;. clicking a summary sentence highlights the text sentences it matched best.
            </p>
          </More>
          <More title="where does it fall short?">
            <p>
              it only reads roughly the first 360 words, which is fine for news (the key facts come first) but not for
              long reports. it learned from news, so emails, papers or stories come out worse. the in-browser version
              writes its summary word by word without trying alternatives, which is a bit more error-prone than the
              version used for the scores in the report: it sometimes mixes up who said what or swaps a number. the
              fact check catches some of these but not all, and it can also flag correct summary sentences that
              paraphrase heavily.
            </p>
          </More>
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
