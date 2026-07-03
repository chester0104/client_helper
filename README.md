# Meridian Assistant

A customer-support chatbot for a small company website. It's a Flask app that
answers questions from a company knowledge base and looks up the signed-in
customer's orders through the Gemini API. Replies stream in as they're generated.

This is a demo project - the company (Meridian Supply Co.), the customer, and the
orders are all made up.

## What it does

- Answers questions from `knowledge.md` (hours, pricing, returns, and so on), and
  says when it doesn't know something instead of guessing.
- Calls functions - `list_orders`, `get_order`, `check_stock`, `start_return` -
  to fetch live order data rather than making it up.
- Stays on topic. It turns down unrelated requests (writing code, homework, etc.)
  and doesn't fall for "ignore your instructions" prompts.
- Has a signed-in account page that reads from the same data the chat tools use,
  so the two never disagree.
- Runs in a resizable corner widget, renders Markdown, and has a light/dark toggle.

## Live demo

https://client-helper.onrender.com

## Screenshots

Order lookup and return - the assistant calls tools for the status, then starts
the return:

![Order lookup and return](docs/return_order_status.gif)

<img src="docs/return_order_status.png" alt="Order lookup and return, final response" width="426" />

Answering from the knowledge base:

![Business hours and return policy](docs/business_hour_return_policy.gif)

<img src="docs/business_hour_return_policy.png" alt="Business hours and return policy, final response" width="426" />

Turning down an off-topic / prompt-injection attempt:

![Refusing an off-topic request](docs/red_teaming.gif)

<img src="docs/red_teaming.png" alt="Refusing an off-topic request, final response" width="426" />

## How it works

The browser posts the conversation to `/chat`. The server calls Gemini with the
system prompt, the knowledge base, and the tool definitions, and streams the
reply back over Server-Sent Events. If Gemini asks for a tool, the server runs it
(`run_tool` in `main.py`), feeds the result back, and continues until the answer
is finished.

- Company facts live in `knowledge.md`.
- Customer and order data live in `account.json`, reached only through tools.
- Scope and refusals are set by the system prompt in `main.py`.

## Running locally

```bash
git clone <your-repo-url>
cd client_helper
pip install -r requirements.txt
cp .env.example .env          # put your Gemini API key in .env
python main.py
```

Open <http://localhost:5000>. Get a free key at <https://aistudio.google.com/apikey>.

## Deploying

There's a `Procfile`, so any host that reads it works. On Render:

1. Push to GitHub.
2. Create a Web Service and connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn main:app --bind 0.0.0.0:$PORT --timeout 120`
5. Set an `API_KEY` environment variable to your Gemini key.

Railway and Fly.io work the same way.

## Layout

```
client_helper/
├── main.py                 # server, tool definitions, streaming chat loop
├── knowledge.md            # company knowledge base (edit to change answers)
├── account.json            # signed-in customer's profile + orders (demo data)
├── requirements.txt
├── Procfile                # start command for production (gunicorn)
├── .env.example            # template for the API key (real .env is git-ignored)
└── templates/
    ├── index.html          # landing page
    ├── account.html        # account page
    └── chat_widget.html    # chat widget: styles, theme, streaming JS
```

## Notes

- Built with Flask, the Google GenAI SDK (Gemini), and plain JavaScript (marked +
  DOMPurify); gunicorn serves it in production.
- Tools only read the signed-in customer's record.
- Chat output is sanitized with DOMPurify before it's inserted into the page.
