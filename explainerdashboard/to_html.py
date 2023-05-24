"""Helper module to define static html outputs"""

__all__ = [
    "add_header",
    "row",
    "rows",
    "fig",
    "card",
    "dashboard_card",
    "card_row",
    "card_rows",
    "title",
    "div",
    "table_from_df",
    "hide",
    "tabs",
    "input",
    "jumbotron",
]


def add_header(html: str, title="explainerdashboard", resize=True) -> str:
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
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.10.2/dist/umd/popper.min.js" integrity="sha384-7+zCNj/IqJ95wo16oMtfsKbZ9ccEh31eOz1HGyDuCQ6wgnyJNSYdrPa03rtR1zdB" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.min.js" integrity="sha384-QJHtvGhmr9XOIpI6YVutG+2QOK9T+ZnN4kzFN1RtK3zEFEIsxhlmWl5/YESvpZ13" crossorigin="anonymous"></script>
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


def row(*cols) -> str:
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
<div class="col">
{col}
</div>
        """
    row += """
</div> 
    """
    return row


def rows(*col_lists) -> str:
    """Turns a list of lists of html snippets into a series of bootstrap rows
    with equally sized columns for each snippet.

    Example:
        to_html.row(
            ["<div>first snippet</div>", "<div>second snippet</div>"],
            ["<div>second row snippet snippet</div>", "<div>second row snippet two</div>"]
        )
    """
    rows = """
<div>
    <div>
    """
    rows += "    </div><div>".join([row(*cols) for cols in col_lists])
    rows += """
    </div>
</div>
    """
    return rows


def fig(fig, include_plotlyjs="cdn", full_html: bool = False) -> str:
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


def card(
    html: str, title: str = None, subtitle: str = None, border: bool = True
) -> str:
    """Wrap to html snippet in a bootstrap card. You can optionally add a title
    and subtitle to the card.
    """
    if title:
        card_header = (
            f"""<div class="card-header"><h3 class="card-title">{title}</h3>"""
        )
        if subtitle:
            card_header += f"""<h6 class="card-subtitle">{subtitle}</h6></div>"""
        else:
            card_header += "</div>"
    else:
        card_header = ""
    return f"""
<div class="card h-100" {'' if border else 'border-0'}>
  {card_header}
  <div class="card-body">
    <div class="w-100">
    {row(html)}
    </div>
  </div>
</div>
"""


def card_row(*cards) -> str:
    """Turns a series of bootstrap into a row with equally sized
    columns for each card.

    Example:
        to_html.card_row('<div class="card">first card</div>", '<div class="card">second snippet</div>')
    """
    row = f"""
<div class="row row-cols-{len(cards)} g-4" style="margin-top: 20px;">
    """
    for card in cards:
        row += f"""
<div class="col">
{card}
</div>
        """
    row += """
</div> 
    """
    return row


def card_rows(*cardrows_list) -> str:
    """Turn a list of lists of bootstrap cards into a series of bootstrap rows
    with cards.

    Example:
        to_html.card_rows(
            [to_html.card("card1"), to_html.card("card2")],
            [to_html.card("card3"), to_html.card("card4")],
        )
    """
    card_decks = [[card_row(*cards)] for cards in cardrows_list]
    return rows(*card_decks)


def dashboard_card(title: str = None, description: str = None, url: str = None) -> str:
    """Generate a dashboard description card for ExplainerHub.
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


def title(title: str) -> str:
    """wrap a title string in div and <H1></H1>"""
    return f"<div><H1>{title}</H1></div>"


def div(html: str) -> str:
    """wrap an html snippet in a <div></div>"""
    return f"<div>{html}</div>"


def table_from_df(df) -> str:
    """Generate a html table from a pandas DataFrame"""

    header_row = "\n".join([f'      <th scope="col">{col}</th>' for col in df.columns])
    body_rows = ""
    for i, row in df.iterrows():
        body_rows += (
            "    <tr>\n"
            + "\n".join(["      <td>" + str(val) + "</td>" for val in row.values])
            + "\n    </tr>\n"
        )

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


def hide(html: str, hide: bool = False) -> str:
    """optionally hide an html snippet (return empty div) if parameter hide=True"""
    if hide:
        return "<div></div>"
    return html


def tabs(tabs_dict: dict) -> str:
    """Generate a series of bootstrap tabs for a dictionary tabs_dict with the
    name of each tab as the dict key and the html contents of the tab as the dict value.
    """
    tabs_id = "myTab"
    html = f'<ul class="nav nav-tabs" id="{tabs_id}" role="tablist">'
    for i, tab_name in enumerate(tabs_dict.keys()):
        tab_id_name = f"{tabs_id}{i}"
        html += f"""
<li class="nav-item" role="presentation">
    <button class="nav-link{'' if i else ' active'}" id="{tab_id_name}-tab"
    data-bs-toggle="tab" type="button" data-bs-target="#{tab_id_name}" role="tab"
    aria-selected="{'false' if i else 'true'}" aria-controls="{tab_id_name}">{tab_name}</button>
</li>
"""

    html += """\n</ul>\n\n"""
    html += f"""<div class="tab-content" id="{tabs_id}Content">\n\n"""

    for i, (tab_name, tab_contents) in enumerate(tabs_dict.items()):
        tab_id_name = f"{tabs_id}{i}"
        html += f"""
<div class="tab-pane {'' if i else 'active'}" id="{tab_id_name}"
     role="tabpanel" aria-labelledby="{tab_id_name}-tab"> {tab_contents}
</div>
"""
    html += "\n</div>"
    # https://getbootstrap.com/docs/5.1/components/navs-tabs/#via-javascript
    html += f"""
<script type="text/javascript">
var triggerTabList = [].slice.call(document.querySelectorAll('#{tabs_id} button'))
triggerTabList.forEach(function (triggerEl) {{
  var tabTrigger = new bootstrap.Tab(triggerEl)

  triggerEl.addEventListener('click', function (event) {{
    event.preventDefault()
    tabTrigger.show()
    window.dispatchEvent(new Event('resize'));
  }})
}})
</script>
"""
    return html


def input(feature: str, value, disabled: bool = False) -> str:
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


def jumbotron(title: str, description: str) -> str:
    """display a bootstrap jumbotron with title and description"""
    # see https://getbootstrap.com/docs/5.0/migration/#jumbotron
    return f"""
    <div class="p-5 mb-4 bg-light rounded-3">
      <div class="container-fluid py-5">
        <h1 class="display-5 fw-bold">{title}</h1>
        <p class="col-md-8 fs-4">{description}</p>
      </div>
    </div>
    """
