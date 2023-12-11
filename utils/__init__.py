import re

###
### Sanitization Functions
###
def sanitize_observation(observation : str) -> str:
    ### Remove disgusting room info formatting
    room_info_formatting_pattern = r"-=[a-zA-Z0-9_\s]*=-"
    alphanumeric_only_pattern = r"^[\W_]+|[\W_]+$"
    quest_move_counter_pattern = r"[0-9]*/[0-9]*"
    carat_removal_pattern = r"\>"
    multiple_newline_reduce = r"\n\n+"
    textworld_graphic_pattern = r"[_]+[\s\S]+\$\$\$\$\$\$\$\s"
    #correct_sentence_pattern = r"\.[a-zA-Z0-9_]+[^$]"
    the_end_replacement_pattern = r"(\s)+\*\*\* The End \*\*\*"

    ### Replace multiple new lines with a single newline
    sanitized_observation = observation

    ### Pattern matching fun
    #sanitized_observation = re.sub(room_info_formatting_pattern, '', sanitized_observation)
    #sanitized_observation = re.sub(quest_move_counter_pattern, '', sanitized_observation)
    #sanitized_observation = re.sub(carat_removal_pattern, '', sanitized_observation)
    #sanitized_observation = re.sub(textworld_graphic_pattern, '', sanitized_observation, count=1, flags=re.MULTILINE)
    sanitized_observation = re.sub(multiple_newline_reduce, '\n', sanitized_observation)
    #sanitized_observation = re.sub(the_end_replacement_pattern, ' You won.', sanitized_observation)

    ### NOTE: Should go after the other operations
    #if enforce_alphanumeric_only:
    #    sanitized_observation = re.sub(alphanumeric_only_pattern, '', sanitized_observation)

    ### Strip to remove any trailing spaces
    sanitized_observation = sanitized_observation.strip()

    return sanitized_observation

def sanitize_response(response : str) -> str:
    sanitized_response = response.strip()
    return sanitized_response
