{ ... }: {
  perSystem = { pkgs, ... }: {
    devShells.bnb = pkgs.mkShell {
      name = "bnb-qlora";
      buildInputs = [ pkgs.python3 pkgs.python3Packages.pip pkgs.git pkgs.cudatoolkit ];
      shellHook = ''
        [ ! -d .venv-bnb ] && python -m venv .venv-bnb
        source .venv-bnb/bin/activate
        pip install -q transformers bitsandbytes accelerate peft trl pandas tqdm
        export CUDA_HOME="${pkgs.cudatoolkit}"
        export LD_LIBRARY_PATH="${pkgs.cudatoolkit}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "=== bnb | transformers+bitsandbytes (QLoRA-ready) ==="
      '';
    };
  };
}
