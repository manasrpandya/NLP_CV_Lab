# run name -> (label, group, params in millions, int8/fp16 size note)
rows = {
    "lead3": ("lead-3", "extractive", 0),
    "textrank3": ("textrank, 3 sentences", "extractive", 0),
    "distilbart-cnn": ("distilbart-cnn-12-6", "pretrained", 306),
    "pegasus-cnndm": ("pegasus-large", "pretrained", 568),
    "bart-large-cnn": ("bart-large-cnn", "pretrained", 406),
    "ours-medium-b4": ("ours, beam 4", "ours", 139),
    "ours-medium-greedy": ("ours, greedy", "ours", 139),
}
main = [
    "lead3",
    "textrank3",
    "distilbart-cnn",
    "pegasus-cnndm",
    "bart-large-cnn",
    "ours-medium-b4",
    "ours-medium-greedy",
]
xs = {
    "xsum-lead1": ("lead-1", "extractive", 0),
    "xsum-lead3": ("lead-3", "extractive", 0),
    "xsum-textrank1": ("textrank, 1 sentence", "extractive", 0),
    "xsum-bart-large": ("bart-large-xsum", "pretrained", 406),
    "xsum-ours": ("ours (bart-base)", "ours", 139),
}
