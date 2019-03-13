from setuptools import setup,find_packages

setup(name='tudou',
      version='2.0.1',
      description='A Chinese NLP tools based Google-bert',
      long_description=open('README.md').read(),
      author='fennuDetudou',
      author_email='upczyxl@163.com',
      packages=find_packages(),
      # include_package_data=True,
      license='MIT',
      package_data={
            # If any package contains *.txt or *.rst files, include them:
            '': ['*.txt', '*.rst','*.pkl','*.json']
      },
      keywords='bert nlp ner NER named entity recognition tensorflow machine learning sentence encoding embedding pos tag sentiment judge',
      )

