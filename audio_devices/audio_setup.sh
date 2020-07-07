#!/bin/bash

# make microphone sink
pacmd load-module module-null-sink sink_name=skype_mic
pacmd update-sink-proplist skype_mic device.description=skype_mic
pacmd update-source-proplist skype_mic.monitor device.description=skype_mic_monitor
# automatically creates "monitor of skype_mic" to select in skype

# make skype speaker sink
pacmd load-module module-null-sink sink_name=skype_out
pacmd update-sink-proplist skype_out device.description=skype_out
pacmd update-source-proplist skype_out.monitor device.description=skype_out_monitor
# automatically creates "monitor of skype_out" to select in program

# configure defaults for python program
pacmd set-default-sink skype_mic
pacmd set-default-source skype_out.monitor

# allow listening to the skype output
pacmd load-module module-loopback latency_msec=1 source=skype_out.monitor sink=0

