{ ... }: {
  perSystem = { pkgs, lib, config, ... }: {
    devShells.uv = pkgs.mkShell {
      name = "uv";
      buildInputs = [ pkgs.uv pkgs.python3 pkgs.git pkgs.stdenv.cc.cc.lib ];
      shellHook = ''
        export UV_PYTHON="${pkgs.python3}/bin/python"
        export UV_PROJECT_ENVIRONMENT=".venv"
        export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        uv sync --frozen 2>/dev/null || uv sync
        source .venv/bin/activate
        echo "=== uv (no CUDA) ==="
      '';
    # low priority — uv-cuda.nix overrides this if imported
    devShells.default = lib.mkDefault config.devShells.uv;
  };
}
