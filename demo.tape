# The README animation, recorded with vhs: https://github.com/charmbracelet/vhs
#
#   brew install vhs
#   MLXSH_HOME=/tmp/mlxsh-demo MLXSH_PORT=41800 MLXSH_GATEWAY_PORT=41880 \
#     vhs demo.tape
#
# The scratch MLXSH_HOME matters: the key shown on screen is a throwaway, and
# the recording cannot disturb whatever is already running.

Output demo.gif

# To check a recording without watching it, add: Output demo.ascii

Set Shell zsh
Set FontSize 15
Set Width 1180
Set Height 760
Set Padding 20
Set Theme "Builtin Dark"
Set TypingSpeed 25ms

Hide
Type "cd /Users/ansnadeem/mlx && setopt interactive_comments && clear" Enter
Show

Type "# serve a model on this Mac" Sleep 300ms Enter Sleep 400ms
Type "./mlxsh.py lm gemma-4-26B" Sleep 300ms Enter Sleep 11s

Type "# one command for a public address: no account, no domain" Sleep 300ms Enter Sleep 400ms
Type "./mlxsh.py tunnel --quick" Sleep 300ms Enter Sleep 16s

Type "URL=$(./mlxsh.py tunnel url); KEY=$(./mlxsh.py gateway key)" Sleep 300ms Enter Sleep 2s

Type "# from anywhere now, and the key is the only lock" Sleep 300ms Enter Sleep 400ms
Type "curl -s $URL/v1/models" Sleep 300ms Enter Sleep 3s

Type `curl -s $URL/v1/models -H "Authorization: Bearer $KEY"` Sleep 300ms Enter Sleep 4s

Type "# a real completion, over the internet" Sleep 300ms Enter Sleep 400ms
Type `curl -s $URL/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d '{"model":"gemma","messages":[{"role":"user","content":"Say hello."}],"max_tokens":200}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"])'` Sleep 300ms Enter Sleep 14s

Type "./mlxsh.py tunnel stop && ./mlxsh.py gateway stop && ./mlxsh.py stop all" Sleep 300ms Enter Sleep 7s
