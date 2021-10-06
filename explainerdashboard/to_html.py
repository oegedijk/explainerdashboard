"""Helper module to define static html outputs"""

__all__ = [
    'add_header',
    'row',
    'rows',
    'fig',
    'card',
    'dashboard_card',
    'card_deck',
    'card_rows',
    'title',
    'div',
    'table_from_df',
    'hide',
    'tabs',
    'input',
    'jumbotron'
]

def add_header(html:str, title="explainerdashboard", resize=True)->str:
    """Turns a html snippet into a full html layout by adding <html>, <head> and <body> tags.
    
    Loads bootstrap css and javascript and triggers a resize event in order to prevent
    plotly figs from overflowing their div containers.

    resize adds a javascript snippet that simulates a window resize in order to
    properly size the plotly figs. (bit of a hack, but it works :)
    """
    full_html = f"""
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
"""
    if resize:
        full_html += """
<script type="text/javascript">
window.dispatchEvent(new Event('resize'));
</script>
        """
    full_html += """
</html>
    """
    return full_html 
    

def row(*cols)->str:
    """Turns a series of html snippets into a bootstrap row with equally sized
    columns for each snippet.

    Example:
        to_html.row("<div>first snippet</div>", "<div>second snippet</div>")
    """
    row = """
<div class="row" style="margin-top: 20px;">
    """
    for col in cols:
        row += f"""
<div class="col-sm">
{col}
</div>
        """
    row += """
</div> 
    """
    return row

def rows(*col_lists)->str:
    """Turns a list of lists of html snippets into a series of bootstrap rows
    with equally sized columns for each snippet.

    Example:
        to_html.row(
            ["<div>first snippet</div>", "<div>second snippet</div>"],
            ["<div>second row snippet snippet</div>", "<div>second row snippet two</div>"]
        )
    """
    rows = [row(*cols) for cols in col_lists]
    return "".join(rows)

def fig(fig, include_plotlyjs='cdn', full_html:bool=False)->str:
    """Returns html for a plotly figure. By default the plotly javascript is not
    included but imported from the plotly cdn, and the full html wrapper is not included.

    Args:
        include_plotlyjs (bool, str): how to import the necessary javascript for the plotly
            fig. Defaults to 'cdn', which means the figure just links to javascript file
            hosted by plotly. If set to True then a 3MB javascript snippet is included.
            For other options check https://plotly.com/python-api-reference/generated/plotly.io.to_html.html
        full_html (bool): include <html>, <head> and <body> tags. Defaults to False.
    """ 
    return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=full_html)

def card(html:str, title:str=None, subtitle:str=None, border=True)->str:
    """Wrap to html snippet in a bootstrap card. You can optionally add a title
    and subtitle to the card.
    """
    if title:
        card_header = f"""<div class="card-header"><h3 class="card-title">{title}</h3>"""
        if subtitle:
            card_header += f"""<h6 class="card-subtitle">{subtitle}</h6></div>"""
        else:
            card_header += "</div>"
    else:
        card_header = ""
    return f"""
<div class="card" {'' if border else 'border-0'}>
  {card_header}
  <div class="card-body">
    <div class="w-100">
    {row(html)}
    </div>
  </div>
</div>
"""

def dashboard_card(title:str=None, description:str=None, url:str=None)->str:
    """Generate a dashboard description car for ExplainerHub. 
    Consists of title, description and url.
    """
    return f"""
<div class="card">
  <div class="card-header">
    <h3 class="card-title">{title}</h3>
  </div>
  <div class="card-body">
    <div class="w-100">
    {description}
    </div>
  </div>
  <div class="card-footer">
    <div class="w-100">
    <a href="{url}" class="stretched-link">Go to dashboard</a>
    </div>
  </div>
</div>
"""


def card_deck(*cards)->str:
    """turn a list of bootstrap cards into an equally spaced card deck.

    Example:
        to_html.card_deck(to_html.card("card1"), to_html.card("card2"))
    """
    cards = list(cards)
    cards = "".join(cards)
    return f"""
<div class="card-deck">
    {cards}
</div>
    """


def card_rows(*card_lists)->str:
    """Turn a list of lists of bootstrap cards into a series of bootstrap rows
    with card decks.
    
    Example:
        to_html.card_rows(
            [to_html.card("card1"), to_html.card("card2")], 
            [to_html.card("card3"), to_html.card("card4")], 
        )
        
        """
    card_decks = [[card_deck(*cards)] for cards in card_lists]
    return rows(*card_decks)


def title(title:str)->str:
    """wrap a title string in div and <H1></H1>"""
    return f"<div><H1>{title}</H1></div>"

    
def div(html:str)->str:
    """wrap an html snippet in a <div></div>"""
    return f'<div>{html}</div>'


def table_from_df(df)->str:
    """Generate a html table from a pandas DataFrame"""

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

def hide(html:str, hide:bool=False)->str:
    """optionally hide an html snippet (return empty div) if parameter hide=True"""
    if hide:
        return "<div></div>"
    return html

def tabs(tabs_dict:dict)->str:
    """Generate a series of bootstrap tabs for a dictionary tabs_dict with the
    name of each tab as the dict key and the html contents of the tab as the dict value.
    """
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


def input(feature:str, value, disabled:bool=False)->str:
    """
    Return a html feature input with a feature name and default value.

    Args:
        feature (str): name of feature
        value (str): default value
        disabled (bool): disable the input. Defaults to False.
    """
    return f"""
<div style="display:flex;flex-direction:column;">
    <label for="{feature}">{feature}</label>
    <input id="{feature}" type="text" value="{value}" name="{feature}" {'disabled' if disabled else ''}>
</div>
    """

def jumbotron(title:str, description:str)->str:
    """display a bootstrap jumbotron with title and description"""
    return f"""
<div class="jumbotron">
    <h1 class="display-4">{title}</h1>
    <hr class="my-2">
    <p class="lead">{description}</p>
</div>

    """