import claude_feature_funcs as claude

procs = [
   ("word_count", word_count),
   ("sentence_count", sentence_count),
   ("readability", readbility_score),
   ("vocabulary", vocab_score),
   ("flesch_reading_ease", claude.flesch_reading_ease),
   ("flesch_kincaid_grade_level", claude.flesch_kincaid_grade_level),
   ("automated_readability_index", claude.automated_readability_index),
   ("gunning_fog_index", claude.gunning_fog_index),
   ("smog_readability", claude.smog_readability),
   ("type_token_ratio", claude.type_token_ratio),
   ("moving_average_ttr", claude.moving_average_ttr),
   ("yule_k_characteristic", claude.yule_k_characteristic),
   ("average_word_length", claude.average_word_length),
   ("syllable_complexity", claude.syllable_complexity),
   ("logical_connector_density", claude.logical_connector_density),
   ("argument_structure_score", claude.argument_structure_score),
   ("coherence_score", claude.coherence_score),
   ("contradiction_detection_score", claude.contradiction_detection_score)
]

