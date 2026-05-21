{ ... }: {
  imports = [ ./vllm-cuda.nix ];

  perSystem = { pkgs, config, ... }: {
    devShells.vllm-cuda-l40 = pkgs.mkShell {
      name = "vllm-cuda-l40";
      buildInputs = [ pkgs.python3 pkgs.python3Packages.pip pkgs.git pkgs.cudatoolkit ];
      shellHook = ''
        [ ! -d .venv ] && python -m venv .venv
        source .venv/bin/activate
        pip install -q vllm openai pandas tqdm
        export CUDA_HOME="${pkgs.cudatoolkit}"
        export LD_LIBRARY_PATH="${pkgs.cudatoolkit}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

        # L40: 48GB VRAM, bf16 no quantization needed
        export VLLM_SERVE_ARGS="--dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 16384"
        echo "=== vllm-cuda-l40 | L40 48GB profile ==="
        echo "Serve: vllm serve meta-llama/Meta-Llama-3-8B-Instruct \$VLLM_SERVE_ARGS"
        echo "Run:   python src/gen.py && python src/eval.py"
      '';
    };
    # Wins over vllm-cuda's mkDefault
    devShells.default = config.devShells.vllm-cuda-l40;
  };
}
