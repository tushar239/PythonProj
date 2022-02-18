"""
response.raise_for_status() returns an HTTPError object if an error has occurred during the process.
It is used for debugging the requests module and is an integral part of Python requests.
Python requests are generally used to fetch the content from a particular resource URI. Whenever we make a request to a specified URI through Python, it returns a response object. Now, this response object would be used to access certain features such as content, headers, etc.
This article revolves around how to check the response.raise_for_status() out of a response object.
"""

# import requests module
import requests

# Making a get request
response = requests.get('https://api.github.com/')

# print response
print(response)  # <Response [200]>

# print check if an error has occurred
print(response.raise_for_status())  # None

# ping an incorrect url
response = requests.get('https://geeksforgeeks.org / naveen/')

# print check if an error has occurred
print(response.raise_for_status())  # socket.gaierror: [Errno 11001] getaddrinfo failed
