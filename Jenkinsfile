pipeline {
  agent any
  options { timestamps() }
  parameters {
    booleanParam(name: 'DEPLOY_TO_K8S', defaultValue: false, description: 'Deploy after tests pass')
  }
  environment {
    PYTHONPATH = "${WORKSPACE}"
  }
  stages {
    stage('Install') {
      steps {
        sh 'python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt'
      }
    }
    stage('Train') {
      steps { sh '. .venv/bin/activate && make train' }
    }
    stage('Test') {
      steps { sh '. .venv/bin/activate && pytest -q' }
    }
    stage('Build') {
      steps { sh 'docker compose build' }
    }
    stage('Deploy') {
      when { expression { return params.DEPLOY_TO_K8S } }
      steps { sh 'kubectl apply -k kubernetes/' }
    }
  }
}
