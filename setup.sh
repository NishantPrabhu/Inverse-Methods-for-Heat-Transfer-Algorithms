mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"me17b084@smail.iitm.ac.in\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
