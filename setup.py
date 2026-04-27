from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[
        RustExtension(
            "filipino_tokenizer._bpe_rust",
            binding=Binding.PyO3,
            debug=False,
        )
    ],
    zip_safe=False,
)
