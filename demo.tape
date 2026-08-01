# A README demo, recorded with vhs: https://github.com/charmbracelet/vhs
#   brew install vhs && vhs demo.tape
#
# Load a small model first so the recording is not mostly waiting:
#   mlxsh pull mlx-community/Qwen3-0.6B-4bit -y

Output demo.gif

Set Shell zsh
Set FontSize 16
Set Width 1100
Set Height 620
Set Padding 18
Set Theme "Builtin Dark"

Type "mlxsh lm qwen3-0.6b"      Sleep 500ms  Enter  Sleep 12s
Type "mlxsh status"             Sleep 500ms  Enter  Sleep 3s
Type "mlxsh ask 'name three uses for a local LLM, one line each'"
Sleep 500ms  Enter  Sleep 12s
Type "mlxsh bench 64"           Sleep 500ms  Enter  Sleep 8s
Type "mlxsh stop"               Sleep 500ms  Enter  Sleep 4s
