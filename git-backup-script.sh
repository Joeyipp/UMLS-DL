#!/bin/bash

timestamp() {
  date +"%c"
}

cd /data/yiph2
/usr/bin/git add -A
/usr/bin/git commit -m "latest update"
/usr/bin/git push -u origin master
timestamp >> git-monitor.txt
