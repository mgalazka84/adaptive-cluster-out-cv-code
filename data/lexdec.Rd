\name{lexdec}
\alias{lexdec}
\docType{data}
\title{Lexical decision latencies for 79 English nouns}
\description{
  Lexical decision latencies elicited from 21 subjects for 79 English concrete nouns, with
  variables linked to subject or word.
}
\usage{data(lexdec)}
\format{
  A data frame with 1659 observations on the following 28 variables.
  \describe{
    \item{\code{Subject}}{a factor for the subjects.}
    \item{\code{RT}}{a numeric vector for logarithmically transformed reaction times.}
    \item{\code{Trial}}{a numeric vector for the rank of the trial in the experimental list.}
    \item{\code{Sex}}{a factor with levels \code{F} (female) and \code{M} (male).}
    \item{\code{NativeLanguage}}{a factor with levels \code{English} and \code{Other}, distinguishing
        between native and nonnative speakers of English.}
    \item{\code{Correct}}{a factor with levels \code{correct} and \code{incorrect} coding whether
        the word was correctly responded to as a word rather than a nonword.}
    \item{\code{PrevType}}{a factor with levels \code{nonword} and \code{word} coding whether the
        item presented at the preceding trial was a word or a nonword.}
    \item{\code{PrevCorrect}}{a factor with levels \code{correct} and \code{incorrect} coding whether
        the preceding item elicited a correct response.}
    \item{\code{Word}}{a factor with 79 words as levels.} 
    \item{\code{Frequency}}{a numeric vector with logarithmically transformed lemma frequencies
        as available in the CELEX lexical database.}
    \item{\code{FamilySize}}{a numeric vector with the log-transformed count of a word's 
        morphological family members.}
    \item{\code{SynsetCount}}{a numeric vector with the log-transformed count of synonym sets 
        in WordNet in which the word is listed.}
    \item{\code{Length}}{a numeric vector for the word's length in letters.}
    \item{\code{Class}}{a factor for the semantic category of the word's referent, with levels 
        \code{animal} and \code{plant}.}
    \item{\code{FreqSingular}}{a numeric vector with the frequency in CELEX of the singular form.}
    \item{\code{FreqPlural}}{a numeric vector with the frequency in CELEX of the plural form.}
    \item{\code{DerivEntropy}}{Shannon's entropy calculated over the frequency 
        distribution of a word's family members.}
    \item{\code{Complex}}{a factor coding morphological complexity with levels \code{complex} and
         \code{simplex}.}
    \item{\code{rInfl}}{a numeric vector for the log of the ratio of the singular to the 
         plural frequency.}
    \item{\code{meanRT}}{a numeric vector for the by-item mean reaction time averaged over subjects.}
    \item{\code{SubjFreq}}{a numeric vector for the by-item mean subjective frequency estimate
         averaged over subjects.}
    \item{\code{meanSize}}{a numeric vector for the by-item mean size rating averaged over subjects.}
    \item{\code{meanWeight}}{a numeric vector for the by-item mean weight rating averaged over subjects.}
    \item{\code{BNCw}}{a numeric vector with the logarithmically transformed frequency in the
         written part of the British National Corpus.}
    \item{\code{BNCc}}{a numeric vector with the logarithmically transformed frequency in the
         context-governed part of the British National Corpus.}
    \item{\code{BNCd}}{a numeric vector  with the logarithmically transformed frequency in the
        demographic part of the British National Corpus.}
    \item{\code{BNCcRatio}}{a numeric vector with the log of the ratio of the (absolute) frequencies
        in the context-governed and written parts of the British National Corpus, normalized
        for the differences in corpus size.}
    \item{\code{BNCdRatio}}{a numeric vector with the log of the ratio of the (absolute) frequencies
        in the demographic and written parts of the British National Corpus, normalized
        for the differences in corpus size.}
  }
}
\source{
Data collected with Jen Hay, University of Canterbury, Christchurch, New Zealand, 2004.
}
\examples{\dontrun{
data(lexdec)
require(lme4)
require(lmerTest)
require(optimx)

lexdec.lmer = lmer(RT ~ 1 + Correct + Trial + PrevType * meanWeight + 
  Frequency + NativeLanguage * Length + (1|Subject) + (1|Word), 
  control=lmerControl(optimizer="optimx",optCtrl=list(method="nlminb")),
  data = lexdec)
summary(lexdec.lmer)

# random slopes

lexdec.lmerA = lmer(RT ~ 1 + Correct + Trial + PrevType * meanWeight + 
  Frequency + NativeLanguage * Length + (Trial|Subject) + (1|Word), 
  control=lmerControl(optimizer="optimx",optCtrl=list(method="nlminb")),
  data = lexdec)
anova(lexdec.lmer, lexdec.lmerA)

lexdec.lmerB = lmer(RT ~ 1 + Correct + Trial + PrevType * meanWeight + 
  Frequency + NativeLanguage * Length + (Trial|Subject) + 
  (Length|Subject) + (1|Word), data = lexdec,
  control=lmerControl(optimizer="optimx",optCtrl=list(method="nlminb")))
anova(lexdec.lmerA, lexdec.lmerB)

# model criticism

qqnorm(resid(lexdec.lmerB))

lexdec.lmerC = lmer(RT ~ 1 + Correct + Trial + PrevType * meanWeight + 
  Frequency + NativeLanguage * Length + 
  (Trial|Subject) + (Length|Subject) + (1|Word), 
  data = lexdec[abs(scale(resid(lexdec.lmerB)))<2,],
  control=lmerControl(optimizer="optimx",optCtrl=list(method="nlminb")))

qqnorm(resid(lexdec.lmerC))

# p values
summary(lexdec.lmerC)

}}
\keyword{datasets}
