{
  description = "treetok: token cluster classifier";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true; # Required for steam-run
      };

      # Wrapper for pixi that uses steam-run (lightweight FHS), needed for
      # compiled deps (xgboost, pyarrow, etc.) to find ld-linux/libc on NixOS.
      pixiWrapped = pkgs.writeShellScriptBin "pixi" ''
        exec ${pkgs.steam-run}/bin/steam-run ${pkgs.pixi}/bin/pixi "$@"
      '';
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pixiWrapped pkgs.git pkgs.just pkgs.zsh ];

        shellHook = ''
          echo "treetok dev environment loaded"
          echo "  Pixi: $(${pkgs.pixi}/bin/pixi --version 2>/dev/null || echo 'available')"
          echo ""
          echo "Run 'pixi install' to set up Python dependencies"
        '';
      };
    };
}
