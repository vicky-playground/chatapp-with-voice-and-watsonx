import requests

def speech_to_text(audio_binary):

    # Set up Watson Speech to Text HTTP Api url
    base_url = 'https://sn-watson-stt.labs.skills.network'
    api_url = base_url+'/speech-to-text/api/v1/recognize'

    # Set up parameters for our HTTP reqeust
    params = {
        'model': 'en-US_Multimedia',
    }

    # Set up the body of our HTTP request
    body = audio_binary

    # Send a HTTP Post request
    response = requests.post(api_url, params=params, data=audio_binary).json()

    # Parse the response to get our transcribed text
    text = 'null'
    while bool(response.get('results')):
        print('speech to text response:', response)
        text = response.get('results').pop().get('alternatives').pop().get('transcript')
        print('recognised text: ', text)
        return text

# To call watsonx's LLM, we need to import the library of IBM Watson Machine Learning
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model

# Define the credentials 
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": 'OwkFRIp2y2Ij7usxHNccwNXb-qwYYiUVGA_duwQRhCvE'
}

# Define the project id
project_id = 'ed0dc754-c5fa-4a17-9b7f-4db34518d2f0'
    
# Specify model_id that will be used for inferencing
model_id = ModelTypes.FLAN_UL2

# Define the model parameters
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024
}

# Define the LLM
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

def text_to_speech(text, voice=""):
    # Set up Watson Text to Speech HTTP Api url
    base_url = 'https://sn-watson-stt.labs.skills.network'
    api_url = base_url + '/text-to-speech/api/v1/synthesize?output=output_text.wav'

    # Adding voice parameter in api_url if the user has selected a preferred voice
    if voice != "" and voice != "default":
        api_url += "&voice=" + voice

    # Set the headers for our HTTP request
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }

    # Set the body of our HTTP request
    json_data = {
        'text': text,
    }

    # Send a HTTP Post reqeust to Watson Text to Speech Service
    response = requests.post(api_url, headers=headers, json=json_data)
    print('text to speech response:', response)
    return response.content

def watsonx_process_message(user_message):
    # Set the prompt for Watsonx API
    prompt = f"""Your name is Watsonx. You act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations.
    Ensure that your answers are clear and complete, and do not include any ellipsis like '[...]'.
    Respond to the query: ```{user_message}```."""
    response_text = model.generate_text(prompt=prompt)
    print("wastonx response:", response_text)
    return response_text


if __name__ == '__main__':
    # Example news to test the model's summarization ability
    news = """
    IBM watsonx is now available to help meet enterprises' AI for business needs
    Today we are announcing that we have begun rolling out IBM watsonx – our enterprise-ready AI and data platform. 
    Previewed at IBM THINK in May, watsonx comprises three products to help organizations accelerate and scale AI – the watsonx.ai studio for new foundation models, generative AI and machine learning (now available); the watsonx.data fit-for-purpose data store, built on an open lakehouse architecture (now available); and the watsonx.governance toolkit to help enable AI workflows to be built with responsibility, transparency and explainability (coming later this year).

    Innovating with clients and partners: 
    Watsonx allows clients and partners to specialize and deploy models for various enterprise use cases or build their own. To date, the platform has been shaped by more than 150 users across industries – from telco to banking – participating in the beta and tech preview programs. See more than 40 logos and some early testimonials shared here. 
    Businesses are excited about the prospect of tapping foundation models and machine learning in one place, with their own data, to accelerate generative AI workloads. “IBM's launch of watsonx was an awakening,” said Sean Im, Samsung SDS America, “and it has inspired us to explore the immense potential of watsonx.ai's generative AI capabilities to deliver unprecedented innovations for our clients." 
    “In an environment where the integration with our systems and seamless interconnection with various software are paramount, watsonx.ai emerges as a compelling solution,” said Atsushi Hasegawa, Chief Engineer, Honda R&D. “Its inherent flexibility and agile deployment capabilities, coupled with a robust commitment to information security, accentuates its appeal.” 
    Citi, a leading global bank serving more than 200 million customers, is delighted to explore the possibilities of generative AI and foundation models within watsonx. “We’re looking at the potential usage of Large Language Models,” said Marc Sabino, Chief Auditor - AI Innovation. Citi. “There are huge possibilities, including connecting your controls to your internal policies.”
    IBM is also working with an expanding ecosystem of partners to co-create and innovate across industries and use cases – from space to sports – including work with NASA to build the first foundation model for analyzing geospatial data and Wimbledon, where watsonx was used to produce tennis commentary.

    Enabling AI builders:
    Today in watsonx.ai, AI builders can leverage models from IBM and from the Hugging Face community for a range of AI development tasks. The models are pre-trained to support a range of Natural Language Processing (NLP) type tasks including question answering, content generation and summarization, text classification and extraction. Future releases will provide access to a greater variety of IBM-trained proprietary foundation models for efficient domain and task specialization. 
    Addressing the global need for foundation models, IBM announced new GPU offerings on IBM Cloud, an AI-tailored infrastructure designed to support enterprise compute-intensive workloads. Later this year, IBM is expected to offer full stack high-performance, flexible, AI-optimized infrastructure, delivered as a service on IBM Cloud, for both training and serving foundation models.

    Down to the data: 
    Designed to help clients overcome pervasive data volume, complexity, cost, and governance challenges when scaling AI workloads, watsonx.data allows users to access their data across cloud and on-premises environments through a single point of entry. 
    And it’s no longer just about the data scientists and engineers ­– watsonx.data empowers non-technical users with self-service access to its own enterprise high-quality, trusted  data in a single collaborative platform, while helping to enable its security and compliance processes through centralized governance and local automated policy enforcement. 
    Later this year, watsonx.data will leverage watsonx.ai foundation models to help simplify and accelerate the way users interact with data, giving them the ability to use natural language to discover, augment, refine, and visualize data and metadata in a conversational user experience. 

    What’s next? Keep an eye on watsonx: 
    Over the next year, watsonx will continue to evolve and we expect to make significant releases. We will be focused on expanding enterprise foundation model use cases beyond NLP and operationalizing 100B+ parameter models for bespoke, targeted use cases – opening the door to broader enterprise adoption. 
    We will also bring to bear the strength of our AI governance capabilities – helping organizations to implement end-to-end lifecycle governance, mitigate risk and manage compliance to the growing AI and industry regulations. AI governance should never be an afterthought, so we encourage our customers begin governance of their ML models and foundation models at the outset. 
    IBM Consulting’s watsonx practice brings expertise in the generative AI technology stack as well as domain and industry experience that can help accelerate clients’ business transformations. 
    In the same way that we established our successful Hybrid Cloud services business built on the Red Hat® OpenShift® platform, IBM Consulting intends to be the leading consulting services provider for watsonx. Businesses are demanding AI that produces accurate and trustworthy results, can scale across clouds, and can be easily adapted to enterprise domains and use cases. 
    Watsonx is designed to help them address those needs. Let’s put AI to work and make the world work better — together."""

    watsonx_process_message(f"Summarize the news: {news}")
