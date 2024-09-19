execute pathogen#infect()
 
call plug#begin('~/.vim/plugged')
 
" Add this line for UltiSnips
Plug 'SirVer/ultisnips'
Plug 'honza/vim-snippets' 
" Initialize plugin system 
Plug 'tpope/vim-pathogen'

" Vim-Airline: A status/tabline plugin
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
 
" Badwolf colorscheme
Plug 'sjl/badwolf'
 
" YouCompleteMe: Code-completion engine
Plug 'ycm-core/YouCompleteMe'
 
call plug#end()
 
" Use <Tab> for triggering snippets
" NOTE: You may need to adjust the mapping if you're using other plugins that use <Tab>
let g:UltiSnipsExpandTrigger="<tab>"
let g:UltiSnipsJumpForwardTrigger="<c-j>"
let g:UltiSnipsJumpBackwardTrigger="<c-k>"
 
" Specify the directory where your snippets will be stored
let g:UltiSnipsSnippetDirectories=['UltiSnips']
set background=dark
syntax on
 
set ts=4
set sts=4
set sw=4
set laststatus=2
set smartindent
set autoindent
set number
set smartcase
set whichwrap+=<,>,[,]
 
let c_no_curly_error=1
let g:badwolf_darkgutter = 1
let g:badwolf_tabline = 2
 
colorscheme badwolf
 
set cursorline
hi CursorLine cterm=NONE ctermbg=NONE ctermfg=NONE guibg=NONE guifg=NONE
hi CursorLineNR cterm=bold
 
nmap <F1> <nop>
imap <F1> <nop>
 
map <F2> <ESC>:w <CR>
imap <F2> <ESC>:w <CR>
map <C-O> :tabnew <bar> :e . <CR>
map <C-W> :q <CR>
imap <C-W> :q <CR>
map <C-T> :tabnew <CR>
noremap <F1> <ESC>:tabprev <CR>
vnoremap <F1> <ESC>:tabprev <CR>
inoremap <F1> <ESC>
autocmd Filetype cpp           map <F9> <ESC>:w <bar> !g++ -std=gnu++17 -O3 -o %:r % && ./%:r <CR>
autocmd Filetype cpp           imap <F9> <ESC>:w <bar> !g++ -std=gnu++17 -O3 -o %:r % && ./%:r <CR>
autocmd Filetype cpp           map <F10> <ESC>:w <bar> !./%:r <CR>
autocmd Filetype cpp           imap <F10> <ESC>:w <bar> !./%:r <CR>
 
autocmd Filetype python        map <F9> <ESC>:w <bar> !python3 % <CR>
autocmd Filetype python        imap <F9> <ESC>:w <bar> !python3 % <CR>
 
autocmd Filetype scala         map <F9> <ESC>:w <bar> !scala -classpath .:/Users/rado/University/ConcurrentProgramming/cp % <CR>
autocmd Filetype scala         imap <F9> <ESC>:w <bar> !scala -classpath .:/Users/rado/University/ConcurrentProgramming/cp % <CR>
 
autocmd Filetype java         map <F9> <ESC>:w <bar> !javac % && java %:r<CR>
autocmd Filetype java         imap <F9> <ESC>:w <bar> !javac % && java %:r<CR>
 
autocmd Filetype haskell       map <F9> <ESC>:w <bar> !ghc % <CR>
autocmd Filetype haskell       imap <F9> <ESC>:w <bar> !ghc % <CR>
autocmd Filetype haskell       map <F8> <ESC>:w <bar> !ghci % <CR>
autocmd Filetype haskell       imap <F8> <ESC>:w <bar> !ghci % <CR>
autocmd Filetype lhaskell       map <F8> <ESC>:w <bar> !ghci % <CR>
autocmd Filetype lhaskell       imap <F8> <ESC>:w <bar> !ghci % <CR>
 
autocmd FileType python        setlocal expandtab
autocmd Filetype cpp           nmap <Tab> mtgg=G't
autocmd Filetype c             nmap <Tab> mtgg=G't
 
set completeopt-=preview
let g:ycm_confirm_extra_conf = 0
let g:ycm_show_diagnostics_ui = 0
 
hi StatusLine ctermbg=black ctermfg=white 
 
"vim-airline
let g:airline_theme='raven'
let g:airline_powerline_fonts = 1
let g:airline#extensions#tabline#enabled = 1
let g:airline#extensions#tabline#show_close_button = 0
let g:airline#extensions#tabline#show_tab_type = 0
let g:airline#extensions#tabline#show_tab_nr = 0
let airline#extensions#tabline#tabs_label = ''
let airline#extensions#tabline#show_splits = 0
let airline#extensions#tabline#show_buffers = 0
"au User AirlineAfterInit  :let g:airline_section_z = airline#section#create(['windowswap', 'obsession', '%3p%%', 'maxlinenr', ' :%3v'])
"au User AirlineAfterInit  :let g:airline_section_z = airline#section#create(['%3p%% %L:%3v'])
au User AirlineAfterInit  :let g:airline_section_z = airline#section#create(['%p%%%#__accent_bold#%{g:airline_symbols.linenr}%l%#__restore__#%#__accent_bold#/%L%#__accent_bold# C:%v%#__restore__#'])
 
"Hide default mode
set noshowmode
 
"Different cursor depending on mode
let &t_SI = "\<Esc>]50;CursorShape=1\x7"       "Vertical bar in insert mode
let &t_EI = "\<Esc>]50;CursorShape=0\x7"       "Block in normal mode
 
" Ensure Backspace works as expected in Insert mode
set backspace=indent,eol,start
 
" General terminal mode settings (usually not needed to modify)
" You can add any additional settings or customizations here
 
" Example: Use a colorscheme that works well in terminal
colorscheme desert
 
" Example: Enable line numbers if needed
set number
set relativenumber 