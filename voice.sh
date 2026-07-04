#!/data/data/com.termux/files/usr/bin/bash
# ============================================================
# voice.sh — Thrivbe Voice Bridge (sox edition)
# Record → send to Hetzner → speak response
#
# Usage:
#   ./voice.sh           → speak query, get spoken response
#   ./voice.sh -t        → text mode (print only, no TTS)
#   ./voice.sh -s 10     → record for 10 seconds (default: 6)
#   ./voice.sh -c "note" → add context to the request
#
# Requirements:
#   pkg install sox curl
# ============================================================

BRIDGE_URL="http://100.114.219.63:8765"
VOICE_BRIDGE_TOKEN="${VOICE_BRIDGE_TOKEN:-}"
RECORD_SECS=6
AUDIO_FILE="$TMPDIR/voice_query.wav"
SPEAK=true
CONTEXT=""

# Parse flags
while getopts "ts:c:" opt; do
  case $opt in
    t) SPEAK=false ;;
    s) RECORD_SECS="$OPTARG" ;;
    c) CONTEXT="$OPTARG" ;;
  esac
done

# ── Check Hetzner reachable ──────────────────────────────────
echo "🔍 Checking Voice Bridge..."
if ! curl -sf --max-time 3 "$BRIDGE_URL/ping" > /dev/null 2>&1; then
  echo "⚠️  Voice Bridge unreachable — are you online and on Tailscale?"
  exit 1
fi

# ── Record audio via sox ─────────────────────────────────────
echo "🎙️  Recording for ${RECORD_SECS}s... speak now!"
rec -r 16000 -c 1 -b 16 -e signed-integer "$AUDIO_FILE" trim 0 "$RECORD_SECS" 2>/dev/null

if [ ! -f "$AUDIO_FILE" ] || [ ! -s "$AUDIO_FILE" ]; then
  echo "❌ Recording failed. Make sure Termux has microphone permission."
  echo "   Go to: Android Settings → Apps → Termux → Permissions → Microphone → Allow"
  exit 1
fi

echo "📡 Sending to Voice Bridge..."

# ── Send to Hetzner, get text response ──────────────────────
RESPONSE=$(curl -sf --max-time 30 \
  -X POST "$BRIDGE_URL/ask" \
  -H "X-Voice-Token: $VOICE_BRIDGE_TOKEN" \
  -F "audio=@$AUDIO_FILE" \
  -F "context=$CONTEXT")

if [ -z "$RESPONSE" ]; then
  echo "❌ No response from Voice Bridge."
  rm -f "$AUDIO_FILE"
  exit 1
fi

echo ""
echo "🤖 $RESPONSE"
echo ""

# ── Speak response via Android TTS ──────────────────────────
if [ "$SPEAK" = true ]; then
  # Use termux-tts-speak if Termux:API available, otherwise espeak
  if command -v termux-tts-speak &>/dev/null; then
    termux-tts-speak "$RESPONSE"
  elif command -v espeak &>/dev/null; then
    espeak "$RESPONSE"
  else
    echo "(Install termux-api or espeak for voice output)"
  fi
fi

rm -f "$AUDIO_FILE"
