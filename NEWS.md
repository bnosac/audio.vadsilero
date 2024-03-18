## CHANGES IN audio.vadsilero VERSION 0.2

- Disable the gradient history recording & set model in evaluation (inference) mode
- Replace package wav with package audio for reading in the wav file
- Default to 1 thread to use as multithreading slows down on CPU
- Use sapply instead of for loop to loop over the windowed samples inside a with_no_grad chunk
- Function silero no longer requires to provide the sample_rate, this is now extracted using audio::load.wave
- Factor out the use of package av to only the examples - internally replaced with package audio

## CHANGES IN audio.vadsilero VERSION 0.1

- Added function Silero to detect voice in audio
- Initial version based on snakers4/silero-vad commit a65732a393366919338fc277019f7a62173e0da6 (Jan 22, 2024)


