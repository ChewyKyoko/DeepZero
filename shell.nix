{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [
    (pkgs.python313.withPackages (ps: with ps; [ torch pandas pyarrow pyyaml matplotlib ]))
  ];
}
