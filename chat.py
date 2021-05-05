import numpy as np
import re
from test_model import encoder_model, decoder_model, num_decoder_tokens, num_encoder_tokens, input_features_dict, target_features_dict, reverse_target_features_dict, max_decoder_seq_length, max_encoder_seq_length

class ChatBot:
    negative_commands = ['no', 'mad', 'angry', 'not', 'don\'t',]
    exit_commands = ['exit', 'leave', 'end', 'bye', 'quit', 'escape', 'stop']

    def start_chat(self):
        chat = input('Hello! Welcome to the cat chatbot. Would you like to discuss with me?')

        #Exit chatbot if user doesn't want to chat
        if chat in self.negative_commands:
            print('Alright... Maybe we can chat about cats later. Bye!')
            return

        #Check if any future lines want to exit and continue with conversation if not
        while not self.make_exit(chat):
            chat = input(self.generate_response(chat) + '\n')

    def make_exit(self, user_input):
        #Check if user wants to exit
        for exit_command in self.exit_commands:
            if exit_command in user_input:
                print('Bye! It was nice talking to you.')
                return True

        return False

    def string_to_matrix(self, user_input):
        #Strip down user_input using regex
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)

        #Create one hot vector with defined shape
        user_input_matrix = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')

        #Check to make sure the token is in the input dictionary
        for timestep, token in enumerate(tokens):
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1. 

        return user_input_matrix

    def generate_response(self, user_input):
        test_input = self.string_to_matrix(user_input)
        states_value = encoder_model.predict(test_input)
    
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))

        # Populate the first token of target sequence with the start token.
        target_seq[0, 0, target_features_dict['<START>']] = 1.
        
        # Sample loop for a batch of sequences
        decoded_sentence = ''
        
        stop_condition = False

        while not stop_condition:
            # Run  decoder model
            output_tokens, hidden_state, cell_state = decoder_model.predict(
                [target_seq] + states_value)
        
            # Choose token with highest probability
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_features_dict[sampled_token_index]
            decoded_sentence += " " + sampled_token
        
            # Exit condition: either hit max length or find stop
            if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True
        
            # Update the target sequence 
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
        
            # Update states
            states_value = [hidden_state, cell_state]
        
        decoded_sentence = decoded_sentence.replace('<START>', '').replace('<END>', '')
        
        return decoded_sentence

catChat = ChatBot()
catChat.start_chat()