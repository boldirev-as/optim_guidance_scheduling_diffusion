import datasets


class HPDv2TestOnly(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {"image": datasets.Image(), "caption": datasets.Value("string")}
            )
        )

    def _split_generators(self, dl_manager):
        url = "https://huggingface.co/datasets/ymhao/HPDv2/resolve/main/test.tar.gz"
        test_tar = dl_manager.download(url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"archive": dl_manager.iter_archive(test_tar)},
            )
        ]

    def _generate_examples(self, archive):
        for idx, (path, f) in enumerate(archive):
            yield idx, {
                "image": {"path": path, "bytes": f.read()},
                "caption": path
            }
