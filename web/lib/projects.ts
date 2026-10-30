// registry of all lab tasks. each project only edits its own entry.
export type st = "live" | "wip" | "todo";

export type proj = {
  n: number;
  slug: string;
  title: string;
  area: "nlp" | "cv" | "multi";
  blurb: string;
  status: st;
};

export const projs: proj[] = [
  {
    n: 1,
    slug: "image-captioning",
    title: "image caption generation",
    area: "multi",
    blurb: "describe an image in one sentence.",
    status: "todo",
  },

  {
    n: 2,
    slug: "translation",
    title: "seq2seq translation",
    area: "nlp",
    blurb: "translate text between languages.",
    status: "todo",
  },

  {
    n: 3,
    slug: "summarization",
    title: "text summarization",
    area: "nlp",
    blurb: "shorten long text with a transformer.",
    status: "todo",
  },

  {
    n: 4,
    slug: "question-answering",
    title: "question answering",
    area: "nlp",
    blurb: "answer questions from a passage.",
    status: "todo",
  },

  {
    n: 5,
    slug: "grammar-correction",
    title: "grammar error correction",
    area: "nlp",
    blurb: "find and fix grammar mistakes.",
    status: "todo",
  },

  {
    n: 6,
    slug: "emotion-classification",
    title: "text emotion classification",
    area: "nlp",
    blurb: "28-way multi-label emotions, running in your browser.",
    status: "wip",
  },

  {
    n: 7,
    slug: "bilateral-denoising",
    title: "bilateral image denoising",
    area: "cv",
    blurb: "classic and learned bilateral filters, live on the gpu.",
    status: "live",
  },

  {
    n: 8,
    slug: "facial-landmarks",
    title: "eye detection & facial landmarks",
    area: "cv",
    blurb: "locate eyes and face keypoints.",
    status: "todo",
  },

  {
    n: 9,
    slug: "activity-recognition",
    title: "human activity recognition",
    area: "cv",
    blurb: "classify actions from video frames.",
    status: "todo",
  },

  {
    n: 10,
    slug: "vehicle-counting",
    title: "vehicle detection & counting",
    area: "cv",
    blurb: "detect and count vehicles in traffic video.",
    status: "todo",
  },

  {
    n: 11,
    slug: "vqa",
    title: "visual question answering",
    area: "multi",
    blurb: "ask questions about an image.",
    status: "todo",
  },

  {
    n: 12,
    slug: "table-extraction",
    title: "table detection & extraction",
    area: "cv",
    blurb: "find tables in documents and pull out the data.",
    status: "todo",
  },

  {
    n: 13,
    slug: "signboard-ocr",
    title: "signboard text recognition & translation",
    area: "multi",
    blurb: "read sign text and translate it.",
    status: "todo",
  },

  {
    n: 14,
    slug: "product-recognition",
    title: "product recognition & description",
    area: "multi",
    blurb: "recognise a product and write a description.",
    status: "todo",
  },

  {
    n: 15,
    slug: "multimodal-sentiment",
    title: "multimodal sentiment analysis",
    area: "multi",
    blurb: "sentiment from text and image together.",
    status: "todo",
  },
];

export const pad = (n: number) => String(n).padStart(2, "0");
export const bySlug = (s: string) => projs.find((p) => p.slug === s);
