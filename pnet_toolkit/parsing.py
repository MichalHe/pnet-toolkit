import xml.etree.ElementTree as ET

from pnet_toolkit.data import PetriNet, PNArc

# Constants
pnet_prefix = '{http://www.pnml.org/version-2009/grammar/pnml}'
arc_tag = f'{pnet_prefix}arc'
inscription_tag = f'{pnet_prefix}inscription'
net_tag = f'{pnet_prefix}net'
name_tag = f'{pnet_prefix}name'
page_tag = f'{pnet_prefix}page'
place_tag = f'{pnet_prefix}place'
text_tag = f'{pnet_prefix}text'
transition_tag = f'{pnet_prefix}transition'


def extract_pn(filepath: str, pn_dom: ET.ElementTree) -> PetriNet:
    """
    Extract the petri net from fiven PNML XML document.
    """
    
    net_name_node = pn_dom.find(f'./{net_tag}/{name_tag}/{text_tag}')
    net_name = net_name_node.text

    page_xpath = f'./{net_tag}/{page_tag}'

    places = set(p.attrib['id'] for p in pn_dom.iterfind(f'{page_xpath}/{place_tag}'))
    transitions = set(t.attrib['id'] for t in pn_dom.iterfind(f'{page_xpath}/{transition_tag}'))
    
    arcs: List[PNArc] = []
    for arc_node in pn_dom.iterfind(f'{page_xpath}/{arc_tag}'):
        arc = PNArc(_id=arc_node.get('id'),
                    source=arc_node.get('source'),
                    target=arc_node.get('target'))
            
        has_weight = False
        inscription_node = arc_node.find(f'./{inscription_tag}')
        if inscription_node:
            text_node = inscription_node.find(f'./{text_tag}')
            if text_node is not None:
                arc.weight = int(text_node.text)
                has_weight = True
        arcs.append(arc)

        if not has_weight and len(arc_node) > 1:
            print(f'Anomaly detected when processing {net_name=}')

    return PetriNet(filepath=filepath,
                    name=net_name,
                    places=places,
                    transitions=transitions,
                    arcs=arcs)
