#!/bin/bash
# Double-click this file in Finder to start the Transcriber app.
cd "$(dirname "$0")"

# Remember this Terminal window by its unique id so it can be closed when the app
# quits (finished windows keep stale tty names, so tty alone could match others)
if [ "$TERM_PROGRAM" = "Apple_Terminal" ]; then
    WINDOW_ID="$(osascript -e "tell application \"Terminal\" to id of first window whose (tty of selected tab is \"$(tty)\" and busy of selected tab is true)" 2>/dev/null)"
fi

pipenv run python app.py

# Close the window just after this script exits, so Terminal doesn't ask to terminate it
if [ -n "$WINDOW_ID" ]; then
    nohup sh -c "sleep 0.5; osascript -e 'tell application \"Terminal\" to close window id $WINDOW_ID'" >/dev/null 2>&1 &
fi
