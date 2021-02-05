import rstr
import morse_talk as mtalk

while True:
    call = rstr.xeger(r'\d?[A-Z]{1,2}\d{1,4}[A-Z]{1,4}') # [A-R][A-R][0-9][0-9]
    if len(call) <= 8:
        print(call, mtalk.encode(call, encoding_type='binary'))
