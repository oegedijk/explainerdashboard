
def add_header(html:str, title="explainerdashboard"):
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<title>{title}</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
</head>
<body>

<div class="container">
{html}
</div>

</body>
<script type="text/javascript">
window.dispatchEvent(new Event('resize'));
</script>
</html>
"""


def row(*cols):
    row = '<div class="row" style="margin-top: 20px;">'
    for col in cols:
        row += '<div class="col-sm">'
        row += col
        row += '</div>'
    row += '</div>'
    return row

def rows(*col_lists):
    rows = [row(*cols) for cols in col_lists]
    rows = "".join(rows)
    return wrap_in_div(rows)

def fig(fig):
    return fig.to_html(include_plotlyjs='cdn', full_html=False)

def card(html, title=None, subtitle=None):
    if title:
        card_header = f"""<div class="card-header"><h3 class="card-title">{title}</h3>"""
        if subtitle:
            card_header += f"""<h6 class="card-subtitle">{subtitle}</h6></div>"""
        else:
            card_header += "</div>"
    else:
        card_header = ""
    return f"""
<div class="card">
  {card_header}
  <div class="card-body">
    <div class="w-100">
    {row(html)}
    </div>
  </div>
</div>
"""


def card_deck(*cards):
    cards = list(cards)
    cards = "".join(cards)
    return f"""
<div class="card-deck">
    {cards}
</div>
    """


def card_rows(*card_lists):
    card_decks = [[card_deck(*cards)] for cards in card_lists]
    return rows(*card_decks)


def title(title):
    return f"<div><H1>{title}</H1></div>"

    
def wrap_in_div(html:str):
    return f'<div>{html}</div>'


def table_from_df(df):
    header_row = '\n'.join([f'      <th scope="col">{col}</th>' for col in df.columns])
    body_rows = ""
    for i, row in df.iterrows():
        body_rows += ('    <tr>\n'+'\n'.join(["      <td>"+str(val)+"</td>" for val in row.values])+'\n    </tr>\n')
        
    table = f"""
<table class="table">
  <thead>
    <tr>
{header_row}
    </tr>
  </thead>
  <tbody>
{body_rows}
  </tbody>
</table>
    """
    return table

def hide(html, hide=False):
    if hide:
        return "<div></div"
    return html

def tabs(tabs_dict):
    html = '<ul class="nav nav-tabs" id="myTab" role="tablist">'
    for i, tab_name in enumerate(tabs_dict.keys()):
        if i == 0:
            html += f"""
    <li class="nav-item">
        <a class="nav-link active" id="{tab_name}-tab" data-toggle="tab" href="#{tab_name}" role="tab" aria-controls="{tab_name}" aria-selected="true">{tab_name}</a>
    </li>\n"""
        else:
            html += f"""
    <li class="nav-item">
        <a class="nav-link" id="{tab_name}-tab" data-toggle="tab" href="#{tab_name}" role="tab" aria-controls="{tab_name}" aria-selected="false">{tab_name}</a>
    </li>\n"""

    html += """\n</ul>\n\n"""
    html += """<div class="tab-content">\n\n"""
    
    for i, (tab_name, tab_contents) in enumerate(tabs_dict.items()):
        if i == 0:
            html += f"""<div class="tab-pane active" id="{tab_name}" role="tabpanel" aria-labelledby="{tab_name}-tab">\n {tab_contents} \n</div>\n"""
        else:
            html += f"""<div class="tab-pane" id="{tab_name}" role="tabpanel" aria-labelledby="{tab_name}-tab">\n {tab_contents} \n</div>\n"""

    html += "\n</div>"

    html += """
<script type="text/javascript">

$('#myTab a').on('click', function (e) {
  e.preventDefault()
  $(this).tab('show')
})

$('a[data-toggle="tab"]').on('shown.bs.tab', function (e) {
    window.dispatchEvent(new Event('resize'));
})

</script>
"""
            
    return html