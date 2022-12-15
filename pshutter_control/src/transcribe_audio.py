#!/usr/bin/env python
from __future__ import division

import re
import sys
import rospy

from google.cloud import speech
from google.oauth2 import service_account

from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

# NOTE: API disabled, enable on google cloud console at https://console.cloud.google.com/apis/library/speech.googleapis.com?project=bim-project-371012

from microphone_stream import MicrophoneStream          

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 1000)  # 10ms

class MoveTranscriber:
    
    def __init__(self):
        rospy.init_node('move_transcriber', anonymous=False)

        self.move_pub = rospy.Publisher('/zzb_hear_move', String, queue_size=1)
        self.rate = rospy.Rate(0.25)

        # TODO: Check if it makes more sense to  sub to zzb_move, look into concurrency issues ... 
        self.sub = rospy.Subscriber('/body_tracking_data', MarkerArray, self.send_move, queue_size=1)

        rospy.spin()

    def send_move(self, msg):

        language_code = "en-US" 
        credentials = service_account.Credentials.from_service_account_file("/Users/kaleb/Downloads/bim-project-371012-6b1b552052a5.json")
        client = speech.SpeechClient(credentials=credentials)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=language_code,
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=True
        )

        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)

            # Now, put the transcription responses to use.
            self.listen_print_loop(responses)


    def listen_print_loop(self, responses):
        """ Iterates through server responses and prints them. """

        num_chars_printed = 0
        for response in responses:
            if not response.results:
                continue

            result = response.results[0] # Best Guess
            if not result.alternatives:
                continue

            # Display the transcription of the top alternative.
            transcript = result.alternatives[0].transcript.lower()

            # Display interim results and  overwrite the previous result
            overwrite_chars = " " * (num_chars_printed - len(transcript))
            pub = ""
            if not result.is_final:
                sys.stdout.write(transcript + overwrite_chars + "\r")
                sys.stdout.flush()

                num_chars_printed = len(transcript)

            else:
                print(transcript + overwrite_chars)
                # print("Transript: " + transcript  + "-----Overwrite Chars" + overwrite_chars)
                if ("zip" in transcript or "is" in transcript or "set" in transcript or "ip" in transcript):
                    # print("Guess: zip")
                    pub = "zip"
                elif ( "ap" in transcript or "op" in transcript or "at" in transcript):
                    # print("Guess: zap")
                    pub = "zap"
                elif( "b" in transcript or  "ing" in transcript):
                    # print("Guess: boing")
                    pub = "boing"
                else:
                    # print("TERM NOT RECORGNIZED")
                    pub = "None"

                self.move_pub.publish(pub)

                # Exit recognition if any of the transcribed phrases could be
                # one of our keywords.
                if re.search(r"\b(exit|quit)\b", transcript, re.I):
                    print("Exiting..")
                    break

                num_chars_printed = 0


if __name__ == "__main__":
    print("STARTING TO LISTEN")
    transcriber = MoveTranscriber()
