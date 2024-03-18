#' @title Voice Activity Detection
#' @description Detect the location of active voice in audio using the Silero VAD model. 
#' It works with .wav audio files with a sample rate of 8 or 16 Khz an can be applied over a window of eiher 32, 64 or 96 milliseconds.
#' @param file the path to an audio file which should be a wav file in 16 bit with mono PCM samples (pcm_s16le codec) with a sampling rate of either 8Khz or 16KHz
#' @param milliseconds integer with the number of milliseconds indicating to compute by this number of milliseconds the VAD signal. Can only be 32, 64 or 96 Defaults to 64.
#' @param sample_rate integer with the sample rate of \code{file}. If not provided, will use package av to extract it.
#' @param threshold numeric indicating if the probability is above this threshold, the segment is detected as voiced. Defaults to 0.5 
#' @return an object of class \code{VAD} which is a list with elements
#' \itemize{
#' \item{file: the path to the file}
#' \item{sample_rate: the sample rate of the audio file in Hz}
#' \item{channels: the number of channels in the audio - as the algorithm requires the audio to be mono this should only be 1}
#' \item{samples: the number of samples in the data}
#' \item{type: the type of VAD model - currently only 'silero'}
#' \item{milliseconds: the provided milliseconds - either by 32, 64 or 96 ms frames}
#' \item{frame_length: the frame length corresponding to the provided milliseconds}
#' \item{vad: a data.frame with columns millisecond, probability, has_voice and vad_segment indicating if the audio contains an active voice signal at that millisecond}
#' \item{vad_segments: a data.frame with columns vad_segment, start, end and has_voice where the start/end values are in seconds}
#' \item{vad_stats: a list with elements n_segments, n_segments_has_voice, n_segments_has_no_voice, seconds_has_voice, seconds_has_no_voice, pct_has_voice indicating the number of segments with voice and the duration of the voice/non-voice in the audio}
#' }
#' @export
#' @examples 
#' file <- system.file(package = "audio.vadsilero", "extdata", "test_wav.wav")
#' vad  <- silero(file, milliseconds = 32)
#' vad
#' vad  <- silero(file, milliseconds = 64)
#' vad
#' vad  <- silero(file, milliseconds = 96)
#' vad
#' vad$vad_segments
#' hist(vad$vad$probability, col = "lightblue", xlab = "Probability voiced")
#' plot(vad$vad$millisecond, vad$vad$probability, type = "l", 
#'      xlab = "Millisecond", ylab = "Probability voiced")
#' 
#' \dontrun{
#' library(av)
#' x <- read_audio_bin(file)
#' plot(seq_along(x) / 16000, x, type = "l")
#' abline(v = vad$vad_segments$start, col = "red", lwd = 2)
#' abline(v = vad$vad_segments$end, col = "blue", lwd = 2)
#' 
#' ##
#' ## If you have audio which is not in mono or another sample rate
#' ## consider using R package av to convert to the desired format
#' av_media_info(file)
#' av_audio_convert(file, output = "audio_pcm_16khz.wav", 
#'                  format = "wav", channels = 1, sample_rate = 16000)
#' vad <- silero("audio_pcm_16khz.wav", milliseconds = 64)
#' }
silero <- function(file, 
                   milliseconds = 64,
                   sample_rate, 
                   threshold = 0.5){
    stopifnot(file.exists(file))
    if(requireNamespace(package = "av", quietly = TRUE)){
        info <- av::av_media_info(file)
        if(info$audio$channels != 1){
            stop(sprintf("%s does not contain audio in mono", file))
        }
        if(missing(sample_rate)){
            sample_rate <- info$audio$sample_rate
        }
    }
    milliseconds <- as.integer(milliseconds)
    stopifnot(milliseconds %in% c(32L, 64L, 96L))
    if(!sample_rate %in% c(8000L, 16000L)){
        stop(sprintf("%s should be in 8000Hz or 16000Hz, not in %s Hz", file, sample_rate))
    }
    model <- SILERO()
    msg <- predict.SILERO(model, file, sample_rate = sample_rate, milliseconds = milliseconds, threshold = threshold)
        
    ## Get groups of sequences of voice/non-voice
    grp <- rle(msg$vad$has_voice)
    msg$type <- "silero"
    msg$vad$vad_segment <- rep(seq_along(grp$lengths), grp$lengths)
    segments <- tapply(X = msg$vad$millisecond, INDEX = msg$vad$vad_segment, FUN = range, simplify = F)   
    segments <- data.frame(vad_segment = as.integer(names(segments)),
                           start = vapply(segments, FUN = function(x) x[1], FUN.VALUE = integer(1), USE.NAMES = FALSE) / 1000,
                           end = vapply(segments, FUN = function(x) x[2], FUN.VALUE = integer(1), USE.NAMES = FALSE) / 1000,
                           has_voice = grp$values)
    msg$vad_segments <- segments
    ## Calculate the percentage of voiced signal
    vad_segments_info <- list(
        n_segments = nrow(segments), 
        n_segments_has_voice = sum(segments$has_voice, na.rm = TRUE), 
        n_segments_has_no_voice = sum(!segments$has_voice, na.rm = TRUE),
        seconds_has_voice = sum((segments$end - segments$start)[segments$has_voice], na.rm = TRUE),
        seconds_has_no_voice = sum((segments$end - segments$start)[!segments$has_voice], na.rm = TRUE))
    vad_segments_info$pct_has_voice = vad_segments_info$seconds_has_voice / (vad_segments_info$seconds_has_voice + vad_segments_info$seconds_has_no_voice)
    msg$vad_stats <- vad_segments_info
    class(msg) <- c("VAD", "silero")
    msg
}

SILERO <- function(){
    autograd_set_grad_mode(FALSE)
    mod <- jit_load(system.file(package = "audio.vadsilero", "models", "silero_vad.jit"))
    mod$eval()
    out <- list(model = mod, version = "v4")
    class(out) <- "SILERO"
    out
}

predict.SILERO <- function(object, newdata, sample_rate, milliseconds, window = milliseconds * (sample_rate / 1000), threshold = 0.5){
    sound       <- audio::load.wave(newdata)
    sample_rate <- attr(sound, which = "rate")
    n_samples   <- length(sound)
    sample_rate <- torch::jit_scalar(as.integer(sample_rate))
    
    if(!sample_rate %in% c(8000, 16000)){
        stop("sample_rate should be 8000 or 16000")
    }
    if(!window %in% c(256, 512, 768, 1024, 1536)){
        stop("Unknown combination of milliseconds and sample_rate")
    }
    
    test        <- torch::torch_tensor(sound)
    
    elements    <- seq.int(from = 1, to = n_samples, by = window)
    out         <- numeric(length = length(elements))
    for(i in seq_along(elements)){
        #cat(i, sep = "\n")
        if((elements[i]+window-1) > n_samples){
            samples <- sound[elements[i]:length(sound)]
            samples <- c(samples, rep(as.numeric(0), times = window - length(samples)))
            samples <- torch::torch_tensor(samples)
            out[i]  <- as.numeric(object$model$forward(samples, sr = sample_rate))    
        }else{
            samples <- test[elements[i]:(elements[i]+window-1)]
            #samples <- torch::torch_tensor(samples)
            #print(str(samples))
            out[i]  <- as.numeric(object$model$forward(samples, sr = sample_rate))    
        }
    }
    sample_rate     <- as.integer(sample_rate)
    vad             <- data.frame(millisecond = elements, probability = out, stringsAsFactors = FALSE)
    vad$has_voice   <- ifelse(vad$probability > threshold, TRUE, FALSE)
    vad$millisecond <- as.integer(vad$millisecond / (sample_rate / 1000))
    msg <- list(
        file= newdata,
        sample_rate = sample_rate,
        channels = 1L,
        samples = n_samples,
        milliseconds = milliseconds,
        frame_length = window,
        vad = vad
    )
    msg
}

#' @export
print.VAD <- function(x, ...){
    cat("Voice Activity Detection", "\n")
    cat("  - file:", x$file, "\n")
    cat("  - sample rate:", x$sample_rate, "\n")
    cat("  - VAD type: ", x$type, ", VAD by milliseconds: ", x$milliseconds, ", VAD frame_length: ", x$frame_length, "\n", sep = "")
    cat("    - Percent of audio containing a voiced signal:", paste(round(100*x$vad_stats$pct_has_voice, digits = 1), "%", sep = ""), "\n")
    cat("    - Seconds voiced:", round(x$vad_stats$seconds_has_voice, digits = 1), "\n")
    cat("    - Seconds unvoiced:", round(x$vad_stats$seconds_has_no_voice, digits = 1), "\n")
}
