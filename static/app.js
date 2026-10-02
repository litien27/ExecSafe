const nameForm = document.querySelector('#name-form');
const welcomeScreen = document.querySelector('#welcome-screen');
const interviewPanel = document.querySelector('#interview-panel');
const form = document.querySelector('#answer-form');
const username = document.querySelector('#username');
const answer = document.querySelector('#answer');
const answerStep = document.querySelector('#answer-step');
const answerLabel = document.querySelector('#answer-label');
const choiceStep = document.querySelector('#choice-step');
const choices = document.querySelector('#choices');
const question = document.querySelector('#question');
const context = document.querySelector('#context');
const conversation = document.querySelector('#conversation');
const errorBox = document.querySelector('#error');
const result = document.querySelector('#result');
const stage = document.querySelector('#stage');
const progress = document.querySelector('#progress');
const status = document.querySelector('#status');
const count = document.querySelector('#count');
const respondentIdentity = document.querySelector('#respondent-identity');
const submitButton = form.querySelector('button[type="submit"]');
let selectedChoice = '';
let phase = 'choice';
let currentQuestionText = question.textContent;

answer.addEventListener('input', () => { count.textContent = `${answer.value.length} / 2000`; });
function addAnswer(text) {
  const item = document.createElement('article');
  item.className = 'message user-message';
  item.innerHTML = `<div class="message-bubble"><span class="speaker">YOU</span><p></p></div>`;
  item.querySelector('p').textContent = text;
  conversation.appendChild(item);
  conversation.scrollTop = conversation.scrollHeight;
}

function addQuestion(text, messageContext = '') {
  const item = document.createElement('article');
  item.className = 'message agent-message';
  item.innerHTML = `<div class="avatar">ES</div><div class="message-bubble"><span class="speaker">EXECSAFE</span><p class="context"></p><p class="question-text"></p></div>`;
  item.querySelector('.context').textContent = messageContext;
  item.querySelector('.context').hidden = !messageContext;
  item.querySelector('.question-text').textContent = text;
  conversation.appendChild(item);
  currentQuestionText = text;
  conversation.scrollTop = conversation.scrollHeight;
}

function setChoices(options) {
  choices.innerHTML = options.map((choice) => `<button type="button" class="choice-button" data-choice="${escapeHtml(choice)}">${escapeHtml(choice)}</button>`).join('');
  choices.querySelectorAll('.choice-button').forEach((button) => {
    button.addEventListener('click', () => {
      selectedChoice = button.dataset.choice;
      choices.querySelectorAll('.choice-button').forEach((item) => item.classList.toggle('selected', item === button));
      form.requestSubmit();
    });
  });
}

function showRespondentIdentity(name) {
  respondentIdentity.textContent = `RESPONDENT / ${name}`;
  respondentIdentity.hidden = false;
}

nameForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const name = username.value.trim();
  if (!name) return;
  welcomeScreen.hidden = true;
  interviewPanel.hidden = false;
  showRespondentIdentity(name);
});

function showChoiceStep(data) {
  phase = 'choice';
  selectedChoice = '';
  choiceStep.hidden = false;
  answerStep.hidden = true;
  answer.required = false;
  submitButton.hidden = true;
  setChoices(data.choices);
  addQuestion(data.fixed_question, data.context);
  document.querySelector('.panel-kicker').textContent = `PRIORITY INTERVIEW / ${data.title}`;
}

function showAnswerStep(data) {
  phase = 'answer';
  choiceStep.hidden = true;
  answerStep.hidden = false;
  answer.required = true;
  submitButton.hidden = false;
  answerLabel.textContent = 'EXPLAIN YOUR DECISION';
  addQuestion(data.ai_question);
  answer.focus();
}

function showElaborationStep(data) {
  phase = 'elaboration';
  choiceStep.hidden = true;
  answerStep.hidden = false;
  answer.required = true;
  submitButton.hidden = false;
  answerLabel.textContent = 'ELABORATE YOUR ANSWER';
  addQuestion(data.elaboration_question);
  answer.focus();
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = answer.value.trim();
  if (phase === 'choice' && !selectedChoice) return;
  if ((phase === 'answer' || phase === 'elaboration') && !text) return;
  errorBox.hidden = true;
  const currentQuestion = currentQuestionText;
  answer.disabled = true;
  submitButton.disabled = true;
  try {
    const response = await fetch('/api/respond', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: currentQuestion, answer: text, choice: selectedChoice, username: username.value.trim()}) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Something went wrong.');
    showRespondentIdentity(username.value.trim());
    if (data.done) {
      addAnswer(text);
      form.hidden = true;
      status.textContent = 'COMPLETE';
      progress.style.width = '100%';
      stage.textContent = 'Three-stage interview complete';
      result.hidden = false;
      result.innerHTML = '<p class="eyebrow">INTERVIEW COMPLETE</p><h2>Thank you for completing the interview.</h2><p>Your responses have been saved. You can review the analysis and chat history from Interview Records.</p>';
    } else if (data.phase === 'elaboration') {
      addAnswer(selectedChoice);
      showElaborationStep(data);
      stage.textContent = `Stage ${data.stage} of 3 / elaboration`;
      progress.style.width = `${((data.stage - 1) * 3 + 1) * 100 / 9}%`;
      answer.value = '';
      count.textContent = '0 / 2000';
      answer.disabled = false;
      submitButton.disabled = false;
    } else if (data.phase === 'answer') {
      addAnswer(data.elaboration || selectedChoice);
      showAnswerStep(data);
      stage.textContent = `Stage ${data.stage} of 3 / follow-up`;
      progress.style.width = `${((data.stage - 1) * 2 + 1) * 100 / 6}%`;
      answer.value = '';
      count.textContent = '0 / 2000';
      answer.disabled = false;
      submitButton.disabled = false;
    } else {
      addAnswer(text);
      showChoiceStep(data);
      stage.textContent = `Stage ${data.stage} of 3`;
      progress.style.width = `${(data.stage - 1) * 100 / 3}%`;
      answer.disabled = false;
      submitButton.disabled = false;
    }
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.hidden = false;
    answer.disabled = false;
    submitButton.disabled = false;
  }
});

document.querySelector('#reset').addEventListener('click', async () => {
  await fetch('/api/reset', {method: 'POST'});
  window.location.reload();
});

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char]));
}

setChoices([...document.querySelectorAll('.choice-button')].map((button) => button.dataset.choice));
